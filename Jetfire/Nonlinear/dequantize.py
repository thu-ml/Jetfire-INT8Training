import torch

import triton
import triton.language as tl

CONST_BLOCK=32

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        block_m, block_n = 64, 64
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
def int8_dequantize_kernel(
                    output_ptr, input_ptr, input_scale_ptr,
                    M, N, SM, SN,
                    input_stride_b, input_stride_0, input_stride_1,
                    s_input_stride_b, s_input_stride_0, s_input_stride_1,
                    output_stride_b, output_stride_0, output_stride_1,  
                    QB: tl.constexpr,
                    BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    
    # Block PID
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)
    NUM_BLOCK_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr + pid_b * input_stride_b,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    scale_input_ptr = tl.make_block_ptr(
        base=input_scale_ptr + pid_b * s_input_stride_b,
        shape=(SM, SN),
        strides=(s_input_stride_0, s_input_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr) # (64, 64)
    input = tl.reshape(input, (BLOCK_SM, QB, BLOCK_SN, QB))

    scale_input = tl.load(scale_input_ptr)
    scale_input = tl.reshape(scale_input, (BLOCK_SM, 1, BLOCK_SN, 1))

    dequantize_output = input * scale_input
    dequantize_output = tl.reshape(dequantize_output, (BLOCK_M, BLOCK_N))
    dequantize_output = dequantize_output.to(tl.float32)

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + pid_b * output_stride_b,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    tl.store(output_block_ptr, dequantize_output)
  
def int8_dequantize(x, s_x, QB):
    if len(x.shape) == 2:
        x_2d = True
        x = x.unsqueeze(0)
        s_x = s_x.unsqueeze(0)
    else:
        x_2d = False
    # print(x.shape)

    # defining the input and output tensor
    BS, M, N = x.shape
    _, SM, SN = s_x.shape
    
    y = torch.empty_like(x, dtype=torch.float32, device="cuda")

    grid = lambda META: (
        BS, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_dequantize_kernel[grid](
        y, x, s_x,
        M, N, SM, SN,
        x.stride(0), x.stride(1), x.stride(2),
        s_x.stride(0), s_x.stride(1), s_x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        QB
    )
    if x_2d:
        y = y.squeeze(0)

    return y

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
                plot_name=f'INT8quantize<BLSZ={CONST_BLOCK}><BS={BS}><SL={SL}>',
                args={'BS': BS, 'SL': SL, 'QB': CONST_BLOCK, 'dtype': torch.float16, 'mode': 'time-consuming'}
            )
        )

@triton.testing.perf_report(
    configs
)

def bench_load_store(BS, SL, CDIM, QB, provider, dtype, mode='forward'): # I only use triton as the provider, and mode when benchmarking
    # create data
    x = torch.randn(BS, SL, CDIM, dtype=dtype).cuda()
    sx = x.abs().max(dim=1)[0] / 127
    bias = None
    
    # triton result
    _qx = x.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    sx = _qx.abs().amax(dim=(3, 4)) / 127
    
    _qx = ((_qx / sx.unsqueeze(3).unsqueeze(4)).round()).to(torch.int8)

    qx = _qx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)

    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':
        def y_fwd(): int8_dequantize(qx, sx, QB)
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

def validity_check(BS=2, SL=64, CDIM=64, QB=CONST_BLOCK, dtype=torch.float16):

    qx = (torch.randn(BS, SL, CDIM, dtype=dtype).cuda() * 254 - 127).to(torch.int8)

    # triton result
    _qx = qx.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    sx = torch.randn(BS, SL // QB, CDIM // QB).cuda().to(torch.float16)
    qx = _qx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)
    _rqx = (_qx * sx.unsqueeze(3).unsqueeze(4))
    rqx = _rqx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)

    output_triton = int8_dequantize(qx, sx, QB)

    print(qx)
    print(sx)
    import IPython
    IPython.embed()

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    validity_check(BS=2, SL=128, CDIM=256, QB=CONST_BLOCK, dtype=torch.float16)
    bench_load_store.run(save_path=f'result/time/multi_quantize_block_quantize/BLSZ=64', print_data=True)