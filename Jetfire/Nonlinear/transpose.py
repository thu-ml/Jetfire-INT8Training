import torch
# 4 block
import triton

import triton.language as tl
from triton.language.extra.cuda import libdevice

'''Quantize Operator'''
'''Input uses 1 * 16 group quantization'''
'''Output uses 1 * 16 group quantization'''
'''The input can be 2D or 3D, but the calculation is performed in 2D'''

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block():
    configs = []
    for nstages in [4, 5, 6]:
        for block_m in [64, 128]:
            for block_n in [64, 128]:
                for nwarps in [8, 16, 32]:
                    configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                                num_stages=nstages, num_warps=nwarps,))
    return configs

@triton.autotune(
    configs=[] + get_configs_io_block(),
    key=['M', 'N',],
)
@triton.jit
def _int8_transpose_kernel(
                    output_ptr, # output
                    input_ptr, # input
                    M, N, # shape
                    input_stride_0, input_stride_1, # input stride
                    output_stride_0, output_stride_1, # output stride
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,): # CUDA block size
    
    # Block PID
    pid = tl.program_id(0)
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

    input = tl.load(input_block_ptr)

    output = tl.trans(input)

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, M),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim1 * BLOCK_N, pid_dim0 * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )

    tl.store(output_block_ptr, output)
  
def int8_transpose(x, transpose_output_2d=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _int8_transpose_kernel[grid](
        y, x,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    # Recover 2D to 3D
    if batched and not transpose_output_2d:
        y = y.reshape(BS, -1, y.shape[-1])

    return y

# I change the dtype of both the input tensor and the output tensor. I use torch.float32, torch.float16, and torch.int8

configs = []
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
            plot_name=f'FP8gelu<SL={SL}>',
            args={'BS': 4, 'SL': SL, 'QB': 16, 'int8type': torch.float8_e4m3fn, 'mode': 'time-consuming'}
        )
    )

@triton.testing.perf_report(
    configs
)

def bench_load_store(BS, SL, CDIM, QB, int8type, provider, mode='forward'): # I only use triton as the provider, and mode when benchmarking
    # create data
    x = torch.randn(BS, SL, CDIM).cuda()
    _qx = x.reshape(BS, SL, CDIM // QB, QB)
    sx = _qx.abs().amax(dim=(3)) / int8_max_value[int8type]
    sx = sx.to(torch.bfloat16)
    _qx = (_qx / sx.unsqueeze(3)).to(int8type)
    qx = _qx.reshape(BS, SL, CDIM)

    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':
        def y_fwd(): int8_transpose(qx, sx, QB)
    if provider == 'torch':
        torch_gelu = torch.nn.SiLU()
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

def validity_check(BS, SL, CDIM, QB, int8type=torch.float8_e4m3fn):
    # create data
    x = torch.randn(BS, SL, CDIM).cuda()
    _qx = x.reshape(BS, SL, CDIM // QB, QB)
    sx = _qx.abs().amax(dim=(3)) / int8_max_value[int8type]
    sx = sx.to(torch.bfloat16)
    _qx = (_qx / sx.unsqueeze(3)).to(int8type)
    qx = _qx.reshape(BS, SL, CDIM)

    # torch result
    torch_silu = torch.nn.SiLU()
    output_torch = torch_silu(x)
    
    # import IPython
    # IPython.embed()

    # triton result
    x_triton, s_triton = int8_transpose(qx, sx, QB)

    _x_triton = x_triton.reshape(BS, SL, CDIM // QB, QB)
    _x_triton = _x_triton.to(torch.float32)
    s_triton = s_triton.unsqueeze(3)
    output_triton = (_x_triton * s_triton).reshape(BS, SL, CDIM)

    import IPython
    IPython.embed()

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    validity_check(BS=4, SL=256, CDIM=512, QB=16, int8type=torch.float8_e4m3fn)
    bench_load_store.run(save_path=f'result/time/multi_quantize_block_quantize/BLSZ=64', print_data=True)