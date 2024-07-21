import torch
import BlockQuantize
import BlockQuantizeCUDA as BQC

torch.set_printoptions(precision=6, sci_mode=False, edgeitems=3, linewidth=1000)
torch.manual_seed(0)

def compute_relative_error(tensor1, tensor2):
    norm_diff = torch.norm(tensor1 - tensor2)
    
    relative_error = norm_diff / torch.norm(tensor1)
    
    return relative_error.item()

def quantize_validation():
    for m, n, k, b in [(128, 256, 32, 32), (512, 512, 64, 32), (1024, 2048, 512, 32)]: # 
        x = torch.rand((m, k, ), dtype=torch.float16, requires_grad=False).cuda()
        w = torch.rand((k, n, ), dtype=torch.float16, requires_grad=False).cuda()

        # x = torch.arange(x.numel()).reshape(x.shape).to(x) / 5000
        # w = torch.arange(w.numel()).reshape(w.shape).to(w) / 5000

        # triton result
        _qx = x.reshape(m // b, b, k // b, b).permute(0, 2, 1, 3)
        _qw = w.reshape(k // b, b, n // b, b).permute(0, 2, 1, 3)
        sx = _qx.abs().amax(dim=(2, 3)) / 127
        sw = _qw.abs().amax(dim=(2, 3)) / 127
        
        _qx = ((_qx / sx.unsqueeze(2).unsqueeze(3)).round()).to(torch.int8)
        _qw = ((_qw / sw.unsqueeze(2).unsqueeze(3)).round()).to(torch.int8)

        _rqx = _qx * sx.unsqueeze(2).unsqueeze(3)
        _rqw = _qw * sw.unsqueeze(2).unsqueeze(3)
        rqx = _rqx.permute(0, 2, 1, 3).reshape(m, k)
        rqw = _rqw.permute(0, 2, 1, 3).reshape(k, n)

        qx = _qx.permute(0, 2, 1, 3).reshape(m, k)
        qw = _qw.permute(0, 2, 1, 3).reshape(k, n)

        # output_igemm = quantized_Ofp_igemm_v1(qx, qw, sx, sw, m, n, k)
        output_igemm, scale_igemm = BQC.igemm_output_int_quantize(qx, qw, sx.t().contiguous(), sw, m, n, k)
        scale_igemm = scale_igemm.to(torch.float16)

        _qoutput_igemm = output_igemm.reshape(m // b, b, n // b, b).permute(0, 2, 1, 3)
        _RQoutput_igemm = (_qoutput_igemm * scale_igemm.unsqueeze(2).unsqueeze(3))
        RQoutput_igemmfp = _RQoutput_igemm.permute(0, 2, 1, 3).reshape(m, n)

        torch.cuda.synchronize()
        
        output_torch = torch.matmul(x.to(torch.float32), w.to(torch.float32))
        Qoutput_torch = torch.matmul(qx.to(torch.float32), qw.to(torch.float32))
        RQoutput_torch = torch.matmul(rqx, rqw)
        

        # print(output_igemm, '\n', output_torch)
        # if not torch.allclose(output_igemm.to(torch.float32), output_torch, rtol=5e-2):
        if not torch.allclose(RQoutput_igemmfp.to(torch.float32), RQoutput_torch.to(torch.float32), rtol=1e-1):
            import IPython
            IPython.embed()

        # import IPython
        # IPython.embed()

        error_rate = compute_relative_error(RQoutput_igemmfp, RQoutput_torch)
        print(f"Relative Error When M = {m}, N = {n}, K = {k}, B = {b}, = {error_rate:.6f}")
        error_rate = compute_relative_error(RQoutput_igemmfp, output_torch)
        print(f"Relative Error When M = {m}, N = {n}, K = {k}, B = {b}, = {error_rate:.6f}")

quantize_validation()

def accumulate_quantize_validation(repeat_time=100):

    m, n, k, b = (512, 512, 64, 32) # 
    x = torch.rand((m, k, ), dtype=torch.float16, requires_grad=False).cuda()
    w = torch.rand((k, n, ), dtype=torch.float16, requires_grad=False).cuda()

    # x = torch.arange(x.numel()).reshape(x.shape).to(x) / 5000
    # w = torch.arange(w.numel()).reshape(w.shape).to(w) / 5000

    # triton result
    _qx = x.reshape(m // b, b, k // b, b).permute(0, 2, 1, 3)
    _qw = w.reshape(k // b, b, n // b, b).permute(0, 2, 1, 3)
    sx = _qx.abs().amax(dim=(2, 3)) / 127
    sw = _qw.abs().amax(dim=(2, 3)) / 127
    
    _qx = ((_qx / sx.unsqueeze(2).unsqueeze(3)).round()).to(torch.int8)
    _qw = ((_qw / sw.unsqueeze(2).unsqueeze(3)).round()).to(torch.int8)

    _rqx = _qx * sx.unsqueeze(2).unsqueeze(3)
    _rqw = _qw * sw.unsqueeze(2).unsqueeze(3)
    rqx = _rqx.permute(0, 2, 1, 3).reshape(m, k)
    rqw = _rqw.permute(0, 2, 1, 3).reshape(k, n)

    qx = _qx.permute(0, 2, 1, 3).reshape(m, k)
    qw = _qw.permute(0, 2, 1, 3).reshape(k, n)

    Accumulate_igemmfp = torch.zeros((m, n), dtype=torch.float32, requires_grad=False).cuda()
    for _ in range(repeat_time):
        # output_igemm = quantized_Ofp_igemm_v1(qx, qw, sx, sw, m, n, k)
        output_igemm, scale_igemm = BQC.igemm_output_int_quantize(qx, qw, sx.t().contiguous(), sw, m, n, k)
        scale_igemm = scale_igemm.to(torch.float16)
        _qoutput_igemm = output_igemm.reshape(m // b, b, n // b, b).permute(0, 2, 1, 3)
        _RQoutput_igemm = (_qoutput_igemm * scale_igemm.unsqueeze(2).unsqueeze(3))
        RQoutput_igemmfp = _RQoutput_igemm.permute(0, 2, 1, 3).reshape(m, n)
        Accumulate_igemmfp += RQoutput_igemmfp
        torch.cuda.synchronize()
    Accumulate_igemmfp /= repeat_time

    output_torch = torch.matmul(x.to(torch.float32), w.to(torch.float32))
    Qoutput_torch = torch.matmul(qx.to(torch.float32), qw.to(torch.float32))
    RQoutput_torch = torch.matmul(rqx, rqw)

    print('-' * 50)
    error_rate = compute_relative_error(RQoutput_igemmfp, RQoutput_torch)
    print(f"Relative Error When M = {m}, N = {n}, K = {k}, B = {b}, = {error_rate:.6f}")
    error_rate = compute_relative_error(Accumulate_igemmfp, RQoutput_torch)
    print(f"Relative Error When M = {m}, N = {n}, K = {k}, B = {b}, = {error_rate:.6f}")

accumulate_quantize_validation()