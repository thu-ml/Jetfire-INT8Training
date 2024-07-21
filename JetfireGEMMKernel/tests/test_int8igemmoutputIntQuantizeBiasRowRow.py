import torch
import BlockQuantize
import BlockQuantizeCUDA as BQC

torch.set_printoptions(precision=6, sci_mode=False, edgeitems=3, linewidth=1000)
torch.manual_seed(0)

def compute_relative_error(tensor1, tensor2):
    norm_diff = torch.norm(tensor1 - tensor2)
    
    relative_error = norm_diff / torch.norm(tensor1)
    
    return relative_error.item()

def find_max_difference_location(tensor1, tensor2):
    diff_tensor = (tensor1 - tensor2) / tensor2

    max_value, max_index = torch.max(diff_tensor.flatten(), dim=0)

    tensor1_value = tensor1.flatten()[max_index]
    tensor2_value = tensor2.flatten()[max_index]

    return max_index.item(), tensor1_value.item(), tensor2_value.item()

def quantize_validation():
    for m, n, k, b in [(128, 256, 32, 32), (512, 1024, 64, 32)]: # 
        qx = (torch.rand((m, k, ), dtype=torch.float16, requires_grad=False).cuda() * 254 - 127).to(torch.int8)
        qw = (torch.rand((k, n, ), dtype=torch.float16, requires_grad=False).cuda() * 254 - 127).to(torch.int8)
        bias = torch.rand((n,), dtype=torch.float32, requires_grad=False).cuda()
        # x = torch.arange(x.numel()).reshape(x.shape).to(x) / 5000
        # w = torch.arange(w.numel()).reshape(w.shape).to(w) / 5000

        # triton result
        _qx = qx.reshape(m // b, b, k // b, b).permute(0, 2, 1, 3)
        _qw = qw.reshape(k // b, b, n // b, b).permute(0, 2, 1, 3)
        sx = (torch.rand(m // b, k // b).cuda() / 10).to(torch.float16)
        sw = (torch.rand(k // b, n // b).cuda() / 10).to(torch.float16)

        _rqx = _qx * sx.unsqueeze(2).unsqueeze(3)
        _rqw = _qw * sw.unsqueeze(2).unsqueeze(3)
        rqx = _rqx.permute(0, 2, 1, 3).reshape(m, k)
        rqw = _rqw.permute(0, 2, 1, 3).reshape(k, n)

        qx = _qx.permute(0, 2, 1, 3).reshape(m, k)
        qw = _qw.permute(0, 2, 1, 3).reshape(k, n)

        biasmax = bias.max()
        # output_igemm = quantized_Ofp_igemm_v1(qx, qw, sx, sw, m, n, k)
        output_igemm, scale_igemm = BQC.igemm_output_int_quantize_bias_rowrow(qx, qw, bias, biasmax, sx.t().contiguous(), sw, m, n, k)
        scale_igemm = scale_igemm.to(torch.float16)
        _qoutput_igemm = output_igemm.reshape(m // b, b, n // b, b).permute(0, 2, 1, 3)
        _RQoutput_igemm = (_qoutput_igemm * scale_igemm.unsqueeze(2).unsqueeze(3))
        RQoutput_igemmfp = _RQoutput_igemm.permute(0, 2, 1, 3).reshape(m, n)

        torch.cuda.synchronize()
        
        Qoutput_torch = torch.matmul(qx.to(torch.float32), qw.to(torch.float32))
        RQoutput_torch = torch.matmul(rqx, rqw) + bias.unsqueeze(0)
        

        # print(output_igemm, '\n', output_torch)
        # if not torch.allclose(output_igemm.to(torch.float32), output_torch, rtol=5e-2):
        if not torch.allclose(RQoutput_igemmfp.to(torch.float32), RQoutput_torch.to(torch.float32), rtol=1e-1, atol=5):
            max_index, tensor1_value, tensor2_value = find_max_difference_location(RQoutput_igemmfp, RQoutput_torch)
            import IPython
            IPython.embed()

        # import IPython
        # IPython.embed()

        error_rate = compute_relative_error(RQoutput_igemmfp, RQoutput_torch)
        print(f"Relative Error When M = {m}, N = {n}, K = {k}, B = {b}, = {error_rate:.6f}")

quantize_validation()
