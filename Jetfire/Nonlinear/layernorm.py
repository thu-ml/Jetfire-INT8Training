"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import torch

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .quantize import _stochastic_rounding
import time
import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

try:
    from dequantize import int8_dequantize
    from quantize import int8_quantize
except:
    from .dequantize import int8_dequantize
    from .quantize import int8_quantize

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block_forward():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        block_m, block_n = 64, 256 # Forward Block can be set to (64, 256), (32, 256). They gives similar performance.
        
        for num_warps in [1, 2, 4, 8, 16]:
            configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                        num_stages=num_stages, num_warps=num_warps,))
    return configs

@triton.autotune(
    configs=[] + get_configs_io_block_forward(),
    key=['M', 'N',],
)
@triton.heuristics({
    'BLOCK_SM': lambda args: args["BLOCK_M"] // args["QB"],
    'BLOCK_SN': lambda args: args["BLOCK_N"] // args["QB"],
})
@triton.jit
def _int8_layer_norm_fwd_fused(
    X,  # pointer to the input
    SX,  # pointer to the input scale
    Y,  # pointer to the output
    SY,  # pointer to the output scale
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,  # number of rows in X
    N,  # number of columns in X
    SM,
    SN,
    QB: tl.constexpr,
    eps,  # epsilon to avoid division by zero
    X_stride_0,
    X_stride_1,
    SX_stride_0,
    SX_stride_1,
    Y_stride_0,
    Y_stride_1,
    SY_stride_0,
    SY_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    # # TYPE 1
    pid_dim0 = tl.program_id(0)
    pid_dim1 = tl.program_id(1)

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(M, N),
        strides=(X_stride_0, X_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # input ptr
    scale_input_ptr = tl.make_block_ptr(
        base=SX,
        shape=(SM, SN),
        strides=(SX_stride_0, SX_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    qx = tl.load(input_block_ptr)
    sx = tl.load(scale_input_ptr)
    qx = tl.reshape(qx, (BLOCK_SM, QB, BLOCK_SN, QB))
    sx = tl.reshape(sx, (BLOCK_SM, 1, BLOCK_SN, 1))
    x = qx * sx
    x = tl.reshape(x, (BLOCK_M, BLOCK_N))

    rows = tl.arange(0, BLOCK_M)
    mean = tl.load(Mean + pid_dim0 * BLOCK_M + rows)
    rstd = tl.load(Rstd + pid_dim0 * BLOCK_M + rows)
    mean = tl.reshape(mean, (BLOCK_M, 1))
    rstd = tl.reshape(rstd, (BLOCK_M, 1))
    # Normalize and apply linear transformation
    cols = tl.arange(0, BLOCK_N)

    x_hat = (x - mean) * rstd

    w = tl.load(W + pid_dim1 * BLOCK_N + cols)
    b = tl.load(B + pid_dim1 * BLOCK_N + cols)

    ln_output = x_hat * w + b
    # Write output

    ln_output = tl.reshape(ln_output, (BLOCK_SM, QB, BLOCK_SN, QB))
    # Quantize Scale calculation
    abs_output = tl.abs(ln_output)
    
    # # Fast Max
    max_val = tl.max(abs_output, axis=1) # (1, 1, M, N)
    max_val = tl.max(max_val, axis=2) # （1， 1， M)
    
    scale_output = max_val / 127.
    # scale_output = tl.reshape(scale_output, (4, 1, 4, 1))
    scale_output = tl.reshape(scale_output, (BLOCK_SM, 1, BLOCK_SN, 1))
    # scale_output = tl.view(scale_output, (2, 1, 2, 1))
    
    # Quantize
    ln_output = ln_output / scale_output
    ln_output = libdevice.llrint(ln_output)
    ln_output = ln_output.to(tl.int8)

    scale_output = tl.reshape(scale_output, (BLOCK_SM, BLOCK_SN))
    ln_output = tl.reshape(ln_output, (BLOCK_M, BLOCK_N))
    scale_output = scale_output.to(SY.type.element_ty)

    output_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(M, N),
        strides=(Y_stride_0, Y_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    scale_output_ptr = tl.make_block_ptr(
        base=SY,
        shape=(SM, SN),
        strides=(SY_stride_0, SY_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(output_block_ptr, ln_output)
    tl.store(scale_output_ptr, scale_output)

@triton.heuristics({
    'BLOCK_SM': lambda args: args["BLOCK_M"] // args["QB"],
    'BLOCK_SN': lambda args: args["BLOCK_N"] // args["QB"],
})
@triton.jit
def _int8_layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             SDX,
                             DY,  # pointer to the output gradient
                             SDY,
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             SX,
                             noise_ptr,
                             W,  # pointer to the weights
                             B,  # pointer to the biases
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             X_stride_0, 
                             X_stride_1,
                             s_X_stride_0, 
                             s_X_stride_1,
                             DY_stride_0, 
                             DY_stride_1,  
                             s_DY_stride_0, 
                             s_DY_stride_1,
                             DX_stride_0, 
                             DX_stride_1,
                             s_DX_stride_0, 
                             s_DX_stride_1,
                             DW_stride_0, 
                             DW_stride_1,  
                             DB_stride_0, 
                             DB_stride_1,  
                             M: tl.constexpr,
                             N: tl.constexpr,  # number of columns in X
                             SM: tl.constexpr,
                             SN: tl.constexpr,
                             QB: tl.constexpr,
                             eps,  # epsilon to avoid division by zero
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,
                             STOCHASTIC: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid_dim0 = tl.program_id(0)
    pid_dim1 = tl.program_id(1)

    # pointers
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(M, N),
        strides=(X_stride_0, X_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # input ptr
    scale_X_ptr = tl.make_block_ptr(
        base=SX,
        shape=(SM, SN),
        strides=(s_X_stride_0, s_X_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )
    
    # pointers
    DY_block_ptr = tl.make_block_ptr(
        base=DY,
        shape=(M, N),
        strides=(DY_stride_0, DY_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # input ptr
    scale_DY_ptr = tl.make_block_ptr(
        base=SDY,
        shape=(SM, SN),
        strides=(s_DY_stride_0, s_DY_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )
        
    # pointers
    DW_block_ptr = tl.make_block_ptr(
        base=DW,
        shape=(M // BLOCK_M, N),
        strides=(DW_stride_0, DW_stride_1),
        offsets=(pid_dim0, pid_dim1 * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0)
    )
        
    # pointers
    DB_block_ptr = tl.make_block_ptr(
        base=DB,
        shape=(M // BLOCK_M, N),
        strides=(DB_stride_0, DB_stride_1),
        offsets=(pid_dim0, pid_dim1 * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0)
    )

    # Offset locks and weights/biases gradient pointer for parallel reduction

    # Load data to SRAM
    x = tl.load(X_block_ptr)
    sx = tl.load(scale_X_ptr)
    dy = tl.load(DY_block_ptr)
    sdy = tl.load(scale_DY_ptr)

    rows = tl.arange(0, BLOCK_M)
    mean = tl.load(Mean + pid_dim0 * BLOCK_M + rows)
    rstd = tl.load(Rstd + pid_dim0 * BLOCK_M + rows)
    mean = tl.reshape(mean, (BLOCK_M, 1)).to(tl.float16)
    rstd = tl.reshape(rstd, (BLOCK_M, 1)).to(tl.float16)

    cols = tl.arange(0, BLOCK_N)
    w = tl.load(W + pid_dim1 * BLOCK_N + cols)
    b = tl.load(B + pid_dim1 * BLOCK_N + cols)

    x = tl.reshape(x, (BLOCK_SM, QB, BLOCK_SN, QB))
    sx = tl.reshape(sx, (BLOCK_SM, 1, BLOCK_SN, 1))
    x = x * sx
    x = tl.reshape(x, (BLOCK_M, BLOCK_N))
    
    dy = tl.reshape(dy, (BLOCK_SM, QB, BLOCK_SN, QB))
    sdy = tl.reshape(sdy, (BLOCK_SM, 1, BLOCK_SN, 1))
    dy = dy * sdy
    dy = tl.reshape(dy, (BLOCK_M, BLOCK_N))
    
    # Compute dx
    xhat = (x - mean) * rstd

    wdy = w * dy
    c1 = tl.sum(xhat * wdy, axis=1) / N
    c2 = tl.sum(wdy, axis=1) / N
    c1 = tl.reshape(c1, (BLOCK_M, 1))
    c2 = tl.reshape(c2, (BLOCK_M, 1))
    dx = (wdy - (xhat * c1 + c2)) * rstd
    dx = dx.to(tl.float16)

    dx = tl.reshape(dx, (BLOCK_SM, QB, BLOCK_SN, QB))
    # Quantize Scale calculation
    abs_dx = tl.abs(dx)
    
    # # Fast Max
    max_val = tl.max(abs_dx, axis=1) # (1, 1, M, N)
    max_val = tl.max(max_val, axis=2) # （1， 1， M)
    
    scale_output = max_val / 127.
    scale_output = tl.reshape(scale_output, (BLOCK_SM, 1, BLOCK_SN, 1))
    
    # Quantize
    dx = tl.fdiv(dx.to(tl.float32), scale_output.to(tl.float32))
    dx = tl.reshape(dx, (BLOCK_M, BLOCK_N))

    if STOCHASTIC:
        noise_block_ptr = tl.make_block_ptr(
            base=noise_ptr,
            shape=(M, N),
            strides=(X_stride_0, X_stride_1),
            offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        noise = tl.load(noise_block_ptr)
        dx = _stochastic_rounding(dx, noise)

    dx = libdevice.llrint(dx)
    dx = dx.to(tl.int8)

    scale_output = tl.reshape(scale_output, (BLOCK_SM, BLOCK_SN))
    scale_output = scale_output.to(SDX.type.element_ty)

    # pointers
    DX_block_ptr = tl.make_block_ptr(
        base=DX,
        shape=(M, N),
        strides=(DX_stride_0, DX_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )    # pointers
    scale_DX_ptr = tl.make_block_ptr(
        base=SDX,
        shape=(SM, SN),
        strides=(s_DX_stride_0, s_DX_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0)
    )

    # Write dx
    tl.store(DX_block_ptr, dx)
    tl.store(scale_DX_ptr, scale_output)

    # Accumulate partial sums for dw/db
    partial_dw = tl.sum((dy * xhat), axis=0).to(w.dtype)
    partial_db = tl.sum(dy, axis=0).to(w.dtype)
    # tl.device_print("pdb", partial_db)

    partial_dw = tl.reshape(partial_dw, (1, BLOCK_N))
    partial_db = tl.reshape(partial_db, (1, BLOCK_N))

    tl.store(DW_block_ptr, partial_dw)
    tl.store(DB_block_ptr, partial_db)

@triton.jit
def _int8_layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

def int8_layernorm_forward(x, sx, weight, mean, rstd, QB, eps=1e-6):
    if len(x.shape) == 2:
        x_2d = True
    else:
        x_2d = False
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        sx = sx.reshape(-1, sx.shape[-1])

    y = torch.empty_like(x, dtype=torch.int8)
    sy = torch.empty_like(sx)
    # reshape input data into 2D tensor
    assert len(x.shape) == 2
    x_arg = x.reshape(-1, x.shape[-1])
    bias = torch.zeros_like(weight)
    M, N = x_arg.shape
    SM, SN = sx.shape
    # Less than 64KB per feature: enqueue fused kernel

    # enqueue kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]),
    )
    _int8_layer_norm_fwd_fused[grid](  #
        x_arg, sx, y, sy, weight, bias, mean, rstd,  #
        M, N, SM, SN, QB, eps,  #
        x_arg.stride(0), x_arg.stride(1),
        sx.stride(0), sx.stride(1),
        y.stride(0), y.stride(1),
        sy.stride(0), sy.stride(1),)

    if not x_2d:
        y = y.reshape(BS, -1, y.shape[-1])
        sy = sy.reshape(BS, -1, sy.shape[-1])

    return y, sy, (mean, rstd)

def int8_layernorm_backward(x, sx, g, sg, w, QB, m, v, stochastic=False, eps=1e-6):
    if len(x.shape) == 2:
        x_2d = True
    else:
        x_2d = False
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        sx = sx.reshape(-1, sx.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        sg = sg.reshape(-1, sg.shape[-1])

    # heuristics for amount of parallel reduction stream for DW/DB
    M, N = x.shape
    SM, SN = sx.shape
    block_m = 32
    block_n = 128
    
    _dw = torch.empty((M // block_m, N), dtype=w.dtype, device=w.device)
    _db = torch.empty((M // block_m, N), dtype=w.dtype, device=w.device)
    dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
    db = torch.empty((N, ), dtype=w.dtype, device=w.device)
    dx = torch.empty((M, N), dtype=torch.int8, device=g.device)
    sdx = torch.empty((SM, SN), dtype=torch.float16, device=g.device)
    b = torch.zeros_like(w)

    if stochastic:
        noise = torch.empty_like(x, dtype=torch.float32).uniform_(-0.5, 0.5)
    else:
        noise = None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]),
    )

    _int8_layer_norm_bwd_dx_fused[grid](  #
        dx, sdx, g, sg, _dw, _db, x, sx, noise, w, b, m, v,  #
        x.stride(0), x.stride(1), 
        sx.stride(0), sx.stride(1), 
        g.stride(0), g.stride(1), 
        sg.stride(0), sg.stride(1), 
        dx.stride(0), dx.stride(1), 
        sdx.stride(0), sdx.stride(1), 
        _dw.stride(0), _dw.stride(1), 
        _db.stride(0), _db.stride(1), 
        M, N, SM, SN, QB, eps, 
        BLOCK_M=block_m, BLOCK_N=block_n, STOCHASTIC=stochastic,
        num_warps=2, num_stages=4)

    grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    # accumulate partial sums in separate kernel
    _int8_layer_norm_bwd_dwdb[grid](
        _dw, _db, dw, db, M // block_m, N,  #
        BLOCK_SIZE_M=64,  #
        BLOCK_SIZE_N=64,
        num_warps=2, num_stages=4)

    if not x_2d:
        dx = dx.reshape(BS, -1, dx.shape[-1])
        sdx = sdx.reshape(BS, -1, sdx.shape[-1])
    return dx, sdx, dw

class INT8LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, sx, mean, rstd, QB, normalized_shape, weight, bias, eps):
        # allocate output
        x = x.to(torch.int8)

        torch.cuda.synchronize()
        forward_start_time = time.time()
        y = torch.empty_like(x, dtype=torch.int8)
        sy = torch.empty_like(sx)
        # reshape input data into 2D tensor
        assert len(x.shape) == 2
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        SM, SN = sx.shape
        # Less than 64KB per feature: enqueue fused kernel

        # enqueue kernel
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]),
        )
        _int8_layer_norm_fwd_fused[grid](  #
            x_arg, sx, y, sy, weight, bias, mean, rstd,  #
            M, N, SM, SN, QB, eps,  #
            x_arg.stride(0), x_arg.stride(1),
            sx.stride(0), sx.stride(1),
            y.stride(0), y.stride(1),
            sy.stride(0), sy.stride(1))
        ctx.save_for_backward(x, sx, weight, bias, mean, rstd)
        ctx.QB = QB
        ctx.eps = eps

        torch.cuda.synchronize()
        forward_end_time = time.time()
        print(f"Forward time at shape {x.shape} = {(forward_end_time - forward_start_time) * 1e3}")

        out = int8_dequantize(y, sy, QB)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, sx, w, b, m, v = ctx.saved_tensors
        QB = ctx.QB
        
        dy, sdy = int8_quantize(dy, QB)

        torch.cuda.synchronize()
        backward_start_time = time.time()

        # heuristics for amount of parallel reduction stream for DW/DB
        M, N = x.shape
        SM, SN = sx.shape
        block_m = 32
        block_n = 128
        
        _dw = torch.empty((M // block_m, N), dtype=w.dtype, device=w.device)
        _db = torch.empty((M // block_m, N), dtype=w.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty((M, N), dtype=torch.int8, device=dy.device)
        sdx = torch.empty((SM, SN), dtype=torch.float16, device=dy.device)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]),
        )

        _int8_layer_norm_bwd_dx_fused[grid](  #
            dx, sdx, dy, sdy, _dw, _db, x, sx, w, b, m, v,  #
            x.stride(0), x.stride(1), 
            sx.stride(0), sx.stride(1), 
            dy.stride(0), dy.stride(1), 
            sdy.stride(0), sdy.stride(1), 
            dx.stride(0), dx.stride(1), 
            sdx.stride(0), sdx.stride(1), 
            _dw.stride(0), _dw.stride(1), 
            _db.stride(0), _db.stride(1), 
            M, N, SM, SN, QB, ctx.eps, 
            BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=2, num_stages=4)

        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _int8_layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, M // block_m, N,  #
            BLOCK_SIZE_M=64,  #
            BLOCK_SIZE_N=64,
            num_warps=2, num_stages=4)
        
        torch.cuda.synchronize()
        forward_end_time = time.time()
        print(f"Backward Time at shape {x.shape} = {(forward_end_time - backward_start_time) * 1e3}")

        return dx, sdx, None, None, None, None, dw, db, None

int8_layer_norm = INT8LayerNorm.apply

def test_int8_layer_norm(M, N, B, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    sx_shape = (M // B, N // B)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    qx = (torch.randn(x_shape, dtype=dtype, device='cuda') * 254 - 127).to(torch.int8).to(torch.float16)
    sx = 5 * torch.randn(sx_shape, dtype=dtype, device='cuda')
    dy = 10 * torch.randn(x_shape, dtype=dtype, device='cuda')

    _qx = qx.reshape(M // B, B, N // B, B).permute(0, 2, 1, 3)
    _rqx = _qx * sx.unsqueeze(2).unsqueeze(3)
    rqx = _rqx.permute(0, 2, 1, 3).reshape(M, N).to(torch.float32)
    mean = rqx.mean(dim=1)
    rstd = 1 / rqx.var(dim=1).sqrt()
    rqx = rqx.to(torch.float16)

    qx.requires_grad_(True)
    rqx.requires_grad_(True)
    sx.requires_grad_(True)
    # forward pass

    y_tri = int8_layer_norm(qx, sx, mean, rstd, B, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(rqx, w_shape, weight, bias, eps).to(dtype)
       
    # _y_tri = y_tri.reshape(M // B, B, N // B, B).permute(0, 2, 1, 3)
    # scale_tri = scale_tri.unsqueeze(2).unsqueeze(3)
    # output_triton = (_y_tri * scale_tri).permute(0, 2, 1, 3).reshape(M, N)

    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, sdx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [qx, sx, weight, bias]]
    d_tri = (dx_tri.reshape(4, 32, 8, 32) * sdx_tri.reshape(4, 1, 8, 1)).reshape(128, 256)
    y_tri.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [rqx, weight, bias]]

    import IPython
    IPython.embed()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[1024, 2048, 4096, 8192], # , 
        line_arg='provider',
        line_vals=['int8', 'torch'],
        line_names=['int8', 'Torch'],
        styles=[('green', '-'), ('orange', '-')],
        ylabel='Time(ms)',
        plot_name='layer-norm-forward',
        args={'M': 8192, 'B': 32, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, B, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    sx_shape = (M // B, N // B)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    qx = (torch.randn(x_shape, dtype=dtype, device='cuda') * 254 - 127).to(torch.int8)
    sx = -2.3 + 0.5 * torch.randn(sx_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    mean = x.mean(dim=1)
    rstd = 1 / x.var(dim=1).sqrt()

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'int8':
        
        def y_fwd():
            return int8_layer_norm(qx, sx, mean, rstd, 32, w_shape, weight, bias, eps)  # noqa: F811, E704

    if provider == 'torch':

        def y_fwd():
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    # backward pass
    if mode == 'backward':

        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    if mode == 'both':
        pass
    return ms, max_ms, min_ms

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200, edgeitems=3)
    test_int8_layer_norm(128, 256, 32, torch.float16)
    bench_layer_norm.run(save_path='.', print_data=True)
