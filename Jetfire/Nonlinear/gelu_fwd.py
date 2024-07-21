import torch
# 4 block
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

try:
    from .utils import random_tensor_generator
    from .common import SCALE_MIN_THRES
except:
    from utils import random_tensor_generator

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [64, 128]:
            for block_n in [64, 128]:
                num_warps = 4 if block_n <= 64 else 8

                configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                            num_stages=num_stages, num_warps=num_warps,))
    return configs

@triton.autotune(
    configs=[] + get_configs_io_block(),
    key=['M', 'N',],
)
@triton.heuristics({
    'BLOCK_SM': lambda args: args["BLOCK_M"] // args["QB"],
    'BLOCK_SN': lambda args: args["BLOCK_N"] // args["QB"],
})
@triton.jit
def int8_gelu_kernel_forward(
                    output_ptr, output_scale_ptr, input_ptr, input_scale_ptr,
                    M, N, SM, SN,
                    input_stride_0, input_stride_1,
                    s_input_stride_0, s_input_stride_1,
                    output_stride_0, output_stride_1,  
                    s_output_stride_0, s_output_stride_1,
                    QB: tl.constexpr, SCALE_MIN_THRES: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,):
    
    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # input ptr
    scale_input_ptr = tl.make_block_ptr(
        base=input_scale_ptr,
        shape=(SM, SN),
        strides=(s_input_stride_0, s_input_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr)
    scale_input = tl.load(scale_input_ptr)

    # Dequantize and GELU calculation
    scale_input = tl.reshape(scale_input, (BLOCK_SM, 1, BLOCK_SN, 1))
    input = tl.reshape(input, (BLOCK_SM, QB, BLOCK_SN, QB))
    input = input * scale_input
    # tl.device_print("input", input)

    cdf = (1. + libdevice.erf(input / libdevice.sqrt(2.))) / 2
    gelu_output = cdf * input

    # Quantize Scale calculation
    abs_output = tl.abs(gelu_output)
    
    # # Fast Max
    max_val = tl.max(abs_output, axis=1)
    max_val = tl.max(max_val, axis=2)
    max_val = max_val + SCALE_MIN_THRES
    
    scale_output = max_val / 127.
    scale_output = tl.reshape(scale_output, (BLOCK_SM, 1, BLOCK_SN, 1))
    
    # Quantize
    gelu_output = tl.fdiv(gelu_output, scale_output)
    gelu_output = libdevice.llrint(gelu_output)
    gelu_output = gelu_output.to(tl.int8)

    scale_output = tl.reshape(scale_output, (BLOCK_SM, BLOCK_SN))
    gelu_output = tl.reshape(gelu_output, (BLOCK_M, BLOCK_N))
    scale_output = scale_output.to(tl.float16)

    # debug
    # gelu_output = input
    # scale_output = scale_input
    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    scale_output_ptr = tl.make_block_ptr(
        base=output_scale_ptr,
        shape=(SM, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(output_block_ptr, gelu_output)
    tl.store(scale_output_ptr, scale_output)
  
def int8_gelu_forward(x, s_x, QB):
    if len(x.shape) == 2:
        x_2d = True
    else:
        x_2d = False
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        s_x = s_x.reshape(-1, s_x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    SM, SN = s_x.shape
    
    y = torch.empty_like(x, dtype=torch.int8)
    s_y = torch.empty_like(s_x, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_gelu_kernel_forward[grid](
        y, s_y, x, s_x,
        M, N, SM, SN,
        x.stride(0), x.stride(1),
        s_x.stride(0), s_x.stride(1),
        y.stride(0), y.stride(1),
        s_y.stride(0), s_y.stride(1),
        QB, SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    if not x_2d:
        y = y.reshape(BS, -1, y.shape[-1])
        s_y = s_y.reshape(BS, -1, s_y.shape[-1])

    return y, s_y

# I change the dtype of both the input tensor and the output tensor. I use torch.float32, torch.float16, and torch.int8

configs = []
for BS in [1, 2, 4, 8]:
    for SL in [8192]:
        configs.append(
            triton.testing.Benchmark( # test different matrix size influence
                x_names=['CDIM'],
                x_vals=[1024, 2048, 4096, 8192],
                line_arg='provider',
                line_vals=["triton", "torch"],
                line_names=['triton', 'torch'],
                styles=[('blue', '-'), ('green', '-')],
                ylabel='time-cost',
                plot_name=f'INT8GELU<BS={BS}><SL={SL}>',
                args={'BS': BS, 'SL': SL, 'QB': 32, 'dtype': torch.float16, 'mode': 'time-consuming'}
            )
        )

@triton.testing.perf_report(
    configs
)

def bench_load_store(BS, SL, CDIM, QB, provider, dtype, mode='forward'): # I only use triton as the provider, and mode when benchmarking
    x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, dtype)

    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':
        def y_fwd(): int8_gelu_forward(qx, sx, QB)
    if provider == 'torch':
        torch_gelu = torch.nn.GELU()
        def y_fwd(): return torch_gelu(x)

    # forward pass
    if mode == 'time-consuming':
        convert_func = lambda ms: ms
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    # backward pass
    if mode == 'gbps':
        convert_func = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    return convert_func(ms), convert_func(max_ms), convert_func(min_ms)

def validity_check(BS=2, SL=64, CDIM=64, QB=32, dtype=torch.float16):

    rqx, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, dtype)
    
    # torch result
    torch_gelu = torch.nn.GELU()
    output_torch = torch_gelu(rqx)

    x_triton, s_triton = int8_gelu_forward(qx, sx, QB)
   
    _x_triton = x_triton.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    s_triton = s_triton.unsqueeze(3).unsqueeze(4)
    output_triton = (_x_triton * s_triton).permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)

    print(output_torch)
    print(output_triton)
    import IPython
    IPython.embed()

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    validity_check(BS=2, SL=128, CDIM=128, QB=32, dtype=torch.float16)
    bench_load_store.run(save_path=f'result/time/multi_quantize_block_gelu_forward/BLSZ=64', print_data=True)