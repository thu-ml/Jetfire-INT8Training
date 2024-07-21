import torch
# 4 block
import triton

import triton.language as tl
from triton.language.extra.cuda import libdevice
from .quantize import _stochastic_rounding

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block():
    configs = []
    for nstages in [4, 5, 6]:
        for block_m in [64,]:
            for block_n in [64,]:
                for nwarps in [8, 16, 32]:
                    configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                                num_stages=nstages, num_warps=nwarps,))
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
def int8_add_Ifp_Ig16_Ofp_Og_kernel(
                    output1_ptr, # output
                    output2_ptr, output2_scale_ptr,
                    input1_ptr, # input
                    input2_ptr, input2_scale_ptr, # input
                    noise_ptr,
                    M, N, SM, SN, QB: tl.constexpr, # shape
                    input1_stride_0, input1_stride_1, # input1 stride
                    input2_stride_0, input2_stride_1, # input2 stride
                    s_input2_stride_0, s_input2_stride_1, # scale of input2 stride
                    output1_stride_0, output1_stride_1, # output stride
                    output2_stride_0, output2_stride_1, # output stride
                    s_output2_stride_0, s_output2_stride_1, # scale of output stride
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,
                    STOCHASTIC: tl.constexpr,):
    
    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # --- The first input --- 
    input1_block_ptr = tl.make_block_ptr(
        base=input1_ptr,
        shape=(M, N),
        strides=(input1_stride_0, input1_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    input1 = tl.load(input1_block_ptr)
    input1 = input1.to(tl.float32)
    input1 = tl.reshape(input1, (BLOCK_SM, QB, BLOCK_SN, QB))

    # --- The second input --- 
    input2_block_ptr = tl.make_block_ptr(
        base=input2_ptr,
        shape=(M, N),
        strides=(input2_stride_0, input2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # input ptr
    scale_input2_ptr = tl.make_block_ptr(
        base=input2_scale_ptr,
        shape=(SM, SN),
        strides=(s_input2_stride_0, s_input2_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    input2 = tl.load(input2_block_ptr)
    scale_input2 = tl.load(scale_input2_ptr)

    input2 = input2.to(tl.float32)
    scale_input2 = scale_input2.to(tl.float32)

    # Dequantize and mul calculation
    scale_input2 = tl.reshape(scale_input2, (BLOCK_SM, 1, BLOCK_SN, 1))
    input2 = tl.reshape(input2, (BLOCK_SM, QB, BLOCK_SN, QB))
    input2 = input2 * scale_input2

    # Actual Calculation of Add
    add_output = input1 + input2

    # Quantize the grad 1 - Scale calculation
    abs_add_output = tl.abs(add_output)
    max_val = tl.max(abs_add_output, axis=1)
    max_val = tl.max(max_val, axis=2)
    scale_output2 = max_val / 127.
    scale_output2 = tl.reshape(scale_output2, (BLOCK_SM, 1, BLOCK_SN, 1))

    # save the fp add output
    fp_add_output = add_output.to(output1_ptr.type.element_ty)
    fp_add_output = tl.reshape(fp_add_output, (BLOCK_M, BLOCK_N))

    # pointers
    output1_block_ptr = tl.make_block_ptr(
        base=output1_ptr,
        shape=(M, N),
        strides=(output1_stride_0, output1_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    tl.store(output1_block_ptr, fp_add_output)

    # Quantize
    add_output = tl.div_rn(add_output, scale_output2)
    add_output = tl.reshape(add_output, (BLOCK_M, BLOCK_N))

    if STOCHASTIC:
        noise_block_ptr = tl.make_block_ptr(
            base=noise_ptr,
            shape=(M, N),
            strides=(input1_stride_0, input1_stride_1),
            offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        noise = tl.load(noise_block_ptr)
        add_output = _stochastic_rounding(add_output, noise)

    add_output = add_output.to(output2_ptr.type.element_ty)

    scale_output2 = scale_output2.to(output2_scale_ptr.type.element_ty)
    scale_output2 = tl.reshape(scale_output2, (BLOCK_SM, BLOCK_SN))

    # pointers
    output2_block_ptr = tl.make_block_ptr(
        base=output2_ptr,
        shape=(M, N),
        strides=(output2_stride_0, output2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    scale_output2_ptr = tl.make_block_ptr(
        base=output2_scale_ptr,
        shape=(SM, SN),
        strides=(s_output2_stride_0, s_output2_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )
    tl.store(output2_block_ptr, add_output)
    tl.store(scale_output2_ptr, scale_output2)
  
def int8_add_Ifp_Ig16_Ofp_Og(x1, x2, s_x2, QB, stochastic=False): # suppose x1 is full precision or BF16
    # Change batched 3D input to 2D
    batched = False
    if len(x1.shape) == 3:
        assert len(s_x2.shape) == 3
        batched = True
        BS = x1.shape[0]
        x1 = x1.reshape(-1, x1.shape[-1])
        x2 = x2.reshape(-1, x2.shape[-1])
        s_x2 = s_x2.reshape(-1, s_x2.shape[-1])

    # defining the input and output tensor
    M, N = x1.shape
    SM, SN = s_x2.shape # assume the shape of quantization block size is always 1 * G
    assert x1.shape == x2.shape
    
    y1 = torch.empty_like(x1, dtype=torch.float32)
    y2 = torch.empty_like(x2, dtype=x2.dtype)
    s_y2 = torch.empty_like(s_x2, dtype=s_x2.dtype)

    if stochastic:
        noise = torch.empty_like(x1, dtype=torch.float32).uniform_(-0.5, 0.5)
    else:
        noise = None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_add_Ifp_Ig16_Ofp_Og_kernel[grid](
        y1, y2, s_y2, x1, x2, s_x2, noise,
        M, N, SM, SN, QB,
        x1.stride(0), x1.stride(1),
        x2.stride(0), x2.stride(1),
        s_x2.stride(0), s_x2.stride(1),
        y1.stride(0), y1.stride(1),
        y2.stride(0), y2.stride(1),
        s_y2.stride(0), s_y2.stride(1),
        STOCHASTIC=stochastic
    )

    # Recover 2D to 3D
    if batched:
        y1 = y1.reshape(BS, -1, y1.shape[-1])
        y2 = y2.reshape(BS, -1, y2.shape[-1])
        s_y2 = s_y2.reshape(BS, -1, s_y2.shape[-1])

    return y1, y2, s_y2

# I change the dtype of both the input tensor and the output tensor. I use torch.float32, torch.float16, and torch.fp8

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
            plot_name=f'mul<SL={SL}>',
            args={'BS': 4, 'SL': SL, 'QB': 16, 'mode': 'time-consuming'}
        )
    )

@triton.testing.perf_report(
    configs
)

def bench_load_store(BS, SL, CDIM, QB, provider, mode='forward'): # I only use triton as the provider, and mode when benchmarking
    # create data
    x1 = torch.randn(BS, SL, CDIM).cuda()

    x2 = torch.randn(BS, SL, CDIM).cuda()
    _qx2 = x2.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    sx2 = _qx2.abs().amax(dim=(3, 4)) / 127
    sx2 = sx2.to(torch.bfloat16)
    _qx2 = (_qx2 / sx2.unsqueeze(3).unsqueeze(4)).to(torch.int8)
    qx2 = _qx2.reshape(BS, SL, CDIM)

    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':
        def y_fwd(): int8_add_Ifp_Ig16_Ofp_Og(x1, qx2, sx2, QB)
    if provider == 'torch':
        def y_fwd(): return x1 + x2

    # forward pass
    if mode == 'time-consuming':
        convert_func = lambda ms: ms
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    # backward pass
    if mode == 'gbps':
        convert_func = lambda ms: 2 * x1.numel() * x1.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    return convert_func(ms), convert_func(max_ms), convert_func(min_ms)

def validity_check(BS, SL, CDIM, QB):
    # create data
    x1 = torch.randn(BS, SL, CDIM).cuda()

    x2 = torch.randn(BS, SL, CDIM).cuda()
    _qx2 = x2.reshape(BS, SL // QB, QB, CDIM // QB, QB).permute(0, 1, 3, 2, 4)
    sx2 = _qx2.abs().amax(dim=(3, 4)) / 127
    sx2 = sx2.to(torch.bfloat16)
    _qx2 = (_qx2 / sx2.unsqueeze(3).unsqueeze(4)).to(torch.int8)
    qx2 = _qx2.permute(0, 1, 3, 2, 4).reshape(BS, SL, CDIM)

    # torch result
    output_torch = x1 + x2
    
    # import IPython
    # IPython.embed()

    # triton result
    x_triton = int8_add_Ifp_Ig16_Ofp_Og(x1, qx2, sx2, QB)

    import IPython
    IPython.embed()

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    validity_check(BS=1, SL=64, CDIM=64, QB=32)
    bench_load_store.run(save_path=f'result/time/multi_quantize_block_mul_forward/BLSZ=64', print_data=True)