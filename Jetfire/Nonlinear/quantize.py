import torch

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

try:
    from .utils import random_tensor_generator
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
def int8_quantize_kernel(
                    output_ptr, output_scale_ptr, input_ptr, noise_ptr,
                    M, N, SM, SN,
                    input_stride_0, input_stride_1,
                    output_stride_0, output_stride_1,
                    s_output_stride_0, s_output_stride_1,
                    QB: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,
                    STOCHASTIC: tl.constexpr,):
    
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
    input = tl.load(input_block_ptr).to(tl.float32)
    input = tl.reshape(input, (BLOCK_SM, QB, BLOCK_SN, QB))
    
    # Quantize Scale calculation
    abs_output = tl.abs(input)
    
    # # Fast Max
    max_val = tl.max(abs_output, axis=1) # (1, 1, M, N)
    max_val = tl.max(max_val, axis=2) # （1， 1， M)
    
    # Slow Max
    # max_val = tl.max(abs_output, axis=(1, 3))
    
    scale_output = max_val / 127.
    scale_output = tl.reshape(scale_output, (BLOCK_SM, 1, BLOCK_SN, 1))
    
    # Quantize
    quantize_output = tl.div_rn(input, scale_output)
    quantize_output = tl.reshape(quantize_output, (BLOCK_M, BLOCK_N))

    if STOCHASTIC:
        noise_block_ptr = tl.make_block_ptr(
            base=noise_ptr,
            shape=(M, N),
            strides=(input_stride_0, input_stride_1),
            offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        noise = tl.load(noise_block_ptr)
        quantize_output = _stochastic_rounding(quantize_output, noise)

    quantize_output = libdevice.llrint(quantize_output)
    quantize_output = quantize_output.to(tl.int8)

    scale_output = tl.reshape(scale_output, (BLOCK_SM, BLOCK_SN))
    scale_output = scale_output.to(output_scale_ptr.type.element_ty)

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

    tl.store(output_block_ptr, quantize_output)
    tl.store(scale_output_ptr, scale_output)

@triton.jit
def _stochastic_rounding(output, noise):
    sign = 1 - 2 * libdevice.signbit(output)
    output = tl.abs(output) + noise

    output = sign * tl.clamp(output, min=-128, max=127)
    
    return output

def int8_quantize(x, QB, stochastic=False):
    if len(x.shape) == 2:
        x_2d = True
    else:
        x_2d = False
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    SM, SN = M // QB, N // QB
    
    y = torch.empty_like(x, dtype=torch.int8, device=x.device)
    s_y = torch.empty((SM, SN), dtype=torch.float16, device=x.device)

    if stochastic:
        noise = torch.empty_like(x, dtype=torch.float32).uniform_(-0.5, 0.5)
    else:
        noise = None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_quantize_kernel[grid](
        y, s_y, x, noise,
        M, N, SM, SN,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        s_y.stride(0), s_y.stride(1),
        QB, STOCHASTIC=stochastic
    )
    if not x_2d:
        y = y.reshape(BS, -1, y.shape[-1])
        s_y = s_y.reshape(BS, -1, s_y.shape[-1])
    
    return y, s_y

# I change the dtype of both the input tensor and the output tensor. I use torch.float32, torch.float16, and torch.int8

configs = []
for BS in [1, 2, 4, 8]:
    for SL in [2048, 4096, ]:
        configs.append(
            triton.testing.Benchmark( # test different matrix size influence
                x_names=['CDIM'],
                x_vals=[1024, 2048, 4096, 6144, 8192],
                line_arg='provider',
                line_vals=["triton", "torch"],
                line_names=['triton', 'torch'],
                styles=[('blue', '-'), ('green', '-')],
                ylabel='time-cost',
                plot_name=f'INT8quantize<BS={BS}><SL={SL}>',
                args={'BS': BS, 'SL': SL, 'QB': 32, 'dtype': torch.float16, 'mode': 'time-consuming'}
            )
        )

@triton.testing.perf_report(
    configs
)

def bench_load_store(BS, SL, CDIM, QB, provider, dtype, mode='forward'): # I only use triton as the provider, and mode when benchmarking
    # create data
    x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, dtype)

    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':
        def y_fwd(): int8_quantize(x, QB)
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

    x_triton, s_triton = int8_quantize(rqx, QB)

    _x_triton = x_triton.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    s_triton = s_triton.unsqueeze(3).unsqueeze(4)
    output_triton = (_x_triton * s_triton).permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)
    
    import IPython
    IPython.embed()

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    validity_check(BS=2, SL=128, CDIM=256, QB=32, dtype=torch.float16)
    bench_load_store.run(save_path=f'result/time/multi_quantize_block_quantize/BLSZ=64', print_data=True)