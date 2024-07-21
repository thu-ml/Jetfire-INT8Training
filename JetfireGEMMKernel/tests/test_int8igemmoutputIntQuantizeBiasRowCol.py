import torch
import BlockQuantize
import BlockQuantizeCUDA as BQC

torch.set_printoptions(precision=5, sci_mode=False, edgeitems=3, linewidth=1000)
torch.manual_seed(0)

def compute_relative_error(tensor1, tensor2):
    norm_diff = torch.norm(tensor1 - tensor2)
    
    relative_error = norm_diff / torch.norm(tensor1)
    
    return relative_error.item()

def quantize_validation():
    for m, n, k, b in [(128, 256, 32, 32), (512, 512, 512, 32)]: # 
        qx = (torch.rand((m, k, ), dtype=torch.float16, requires_grad=False).cuda() * 254 - 127).to(torch.int8)
        qw = (torch.rand((n, k, ), dtype=torch.float16, requires_grad=False).cuda() * 254 - 127).to(torch.int8)
        bias = torch.rand((n,), dtype=torch.float32, requires_grad=False).cuda()
        # x = torch.arange(x.numel()).reshape(x.shape).to(x) / 5000
        # w = torch.arange(w.numel()).reshape(w.shape).to(w) / 5000

        # triton result
        _qx = qx.reshape(m // b, b, k // b, b).permute(0, 2, 1, 3)
        _qw = qw.reshape(n // b, b, k // b, b).permute(0, 2, 1, 3)
        sx = (torch.rand(m // b, k // b).cuda() / 10).to(torch.float16)
        sw = (torch.rand(n // b, k // b).cuda() / 10).to(torch.float16)

        _rqx = _qx * sx.unsqueeze(2).unsqueeze(3)
        _rqw = _qw * sw.unsqueeze(2).unsqueeze(3)
        rqx = _rqx.permute(0, 2, 1, 3).reshape(m, k)
        rqw = _rqw.permute(0, 2, 1, 3).reshape(n, k)

        qx = _qx.permute(0, 2, 1, 3).reshape(m, k)
        qw = _qw.permute(0, 2, 1, 3).reshape(n, k)

        biasmax = bias.max()
        qwt = qw.t()
        # output_igemm = quantized_Ofp_igemm_v1(qx, qw, sx, sw, m, n, k)
        output_igemm, scale_igemm = BQC.igemm_output_int_quantize_bias_rowcol(qx, qwt, bias, biasmax, sx.t().contiguous(), sw.t().contiguous(), m, n, k)

        scale_igemm = scale_igemm.to(torch.float16)
        _qoutput_igemm = output_igemm.reshape(m // b, b, n // b, b).permute(0, 2, 1, 3)
        _RQoutput_igemm = (_qoutput_igemm * scale_igemm.unsqueeze(2).unsqueeze(3))
        RQoutput_igemmfp = _RQoutput_igemm.permute(0, 2, 1, 3).reshape(m, n)

        torch.cuda.synchronize()

        Qoutput_torch = torch.matmul(qx.to(torch.float32), qw.to(torch.float32).t())
        RQoutput_torch = torch.matmul(rqx, rqw.t()) + bias

        # print(output_igemm, '\n', output_torch)
        # if not torch.allclose(output_igemm.to(torch.float32), output_torch, rtol=5e-2):
        if not torch.allclose(RQoutput_igemmfp.to(torch.float32), RQoutput_torch.to(torch.float32), rtol=1e-1, atol=5):
            import IPython
            IPython.embed()

        # import IPython
        # IPython.embed()

        error_rate = compute_relative_error(RQoutput_igemmfp, RQoutput_torch)
        print(f"Relative Error When M = {m}, N = {n}, K = {k}, B = {b}, = {error_rate:.6f}")
        # error_rate = compute_relative_error(RQoutput_igemmfp, output_torch)
        # print(f"Relative Error When M = {m}, N = {n}, K = {k}, B = {b}, = {error_rate:.6f}")

quantize_validation()
