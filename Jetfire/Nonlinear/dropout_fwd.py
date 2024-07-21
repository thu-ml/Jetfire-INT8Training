import torch
# 4 block
import triton
import triton.language as tl

CONST_BLOCK=32 # here is B for quant group size ?

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
def int8_dropout_kernel_forward(
                    output_ptr, output_scale_ptr, input_ptr, input_scale_ptr,
                    mask_ptr, p_ptr,
                    M, N, SM, SN,
                    input_stride_b, input_stride_0, input_stride_1,
                    s_input_stride_b, s_input_stride_0, s_input_stride_1,
                    output_stride_b, output_stride_0, output_stride_1,  
                    s_output_stride_b, s_output_stride_0, s_output_stride_1,
                    mask_stride_b, mask_stride_0, mask_stride_1,
                    QB: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,):
    
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

    # input ptr
    scale_input_ptr = tl.make_block_ptr(
        base=input_scale_ptr + pid_b * s_input_stride_b,
        shape=(SM, SN),
        strides=(s_input_stride_0, s_input_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )
    
    mask_block_ptr = tl.make_block_ptr(
        base=mask_ptr + pid_b * mask_stride_b,
        shape=(M, N),
        strides=(mask_stride_0, mask_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    input = tl.load(input_block_ptr)
    scale_input = tl.load(scale_input_ptr)
    mask = tl.load(mask_block_ptr)
    p = tl.load(p_ptr)

    # Dequantize and dropout calculation
    # scale_input = tl.reshape(scale_input, (BLOCK_SM, 1, BLOCK_SN, 1))
    # input = tl.reshape(input, (BLOCK_SM, BLOCK_M//BLOCK_SM, BLOCK_SN, BLOCK_N//BLOCK_SN))
    
    dropout_output = input * mask
    scale_output = scale_input / (1-p)
    
    # scale_output = tl.reshape(scale_output, (4, 1, 4, 1))
    # scale_output = tl.reshape(scale_output, (BLOCK_SM, 1, BLOCK_SN, 1))
    dropout_output = dropout_output.to(tl.int8)
    scale_output = scale_output.to(tl.float16)

    # debug
    # dropout_output = input
    # scale_output = scale_input
    
    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + pid_b * output_stride_b,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    scale_output_ptr = tl.make_block_ptr(
        base=output_scale_ptr + pid_b * s_output_stride_b,
        shape=(SM, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(output_block_ptr, dropout_output)
    tl.store(scale_output_ptr, scale_output)
  
def int8_dropout_forward(x, s_x, mask, p, QB):
    # defining the input and output tensor
    BS, M, N = x.shape
    _, SM, SN = s_x.shape
    
    y = torch.empty_like(x, dtype=torch.int8)
    s_y = torch.empty_like(s_x, dtype=torch.float16)

    grid = lambda META: (
        BS, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_dropout_kernel_forward[grid](
        y, s_y, x, s_x,
        mask, p,
        M, N, SM, SN,
        x.stride(0), x.stride(1), x.stride(2),
        s_x.stride(0), s_x.stride(1), s_x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        s_y.stride(0), s_y.stride(1), s_y.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        QB
    )
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
                plot_name=f'INT8dropout<BLSZ={CONST_BLOCK}><BS={BS}><SL={SL}>',
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

    p = torch.tensor([0.2], dtype=dtype, ).cuda()
    mask = torch.ones_like(x).bernoulli_(1 - p).to(torch.bool)
    
    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':
        def y_fwd(): int8_dropout_forward(qx, sx, mask, p, QB)
    if provider == 'torch':
        torch_dropout = torch.nn.Dropout()
        def y_fwd(): return torch_dropout(x)

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

    # x = torch.randn(BS, SL, CDIM, dtype=dtype).cuda()
    x = torch.ones(BS, SL, CDIM, dtype=dtype, ).cuda()
    p = torch.tensor([0.2], dtype=dtype, ).cuda()
    mask = torch.ones_like(x).bernoulli_(1 - p).to(torch.bool)
    
    sx = x.abs().max(dim=1)[0] / 127
    bias = None
    
    # triton result
    _qx = x.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    sx = _qx.abs().amax(dim=(3, 4)) / 127
    
    _qx = ((_qx / sx.unsqueeze(3).unsqueeze(4)).round()).to(torch.int8)

    qx = _qx.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)
    
    print(qx.shape, sx.shape)
    # exit()

    x_triton, s_triton = int8_dropout_forward(qx, sx, mask, p, QB)
    print(x_triton[0])
   
    _x_triton = x_triton.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    s_triton = s_triton.unsqueeze(3).unsqueeze(4)
    output_triton = (_x_triton * s_triton).permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)
    
    # print(qx)
    # print(sx)
    # import IPython
    # IPython.embed()
    
    print(output_triton)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    validity_check(BS=2, SL=128, CDIM=64, QB=CONST_BLOCK, dtype=torch.float16)
    bench_load_store.run(save_path=f'result/time/multi_quantize_block_dropout/BLSZ=64', print_data=True)