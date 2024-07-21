import torch
import BlockQuantize
import BlockQuantizeCUDA as BQC

torch.set_printoptions(precision=6, sci_mode=False, edgeitems=3, linewidth=1000)
torch.manual_seed(0)

def compute_relative_error(tensor1, tensor2):
    norm_diff = torch.norm(tensor1 - tensor2)
    
    relative_error = norm_diff / torch.norm(tensor1)
    
    return relative_error.item()

dtype = torch.int8
def validation():
    for m, n, k in [(128, 256, 32), (512, 512, 64)]: # , (512, 512, 64)
        a = torch.rand((m, k, ), dtype=torch.float32, requires_grad=False).cuda() * 10
        b = torch.rand((k, n, ), dtype=torch.float32, requires_grad=False).cuda() * 10
        a, b = a.to(dtype), b.to(dtype)

        output_igemm = BQC.igemm_basic_int8_gemm(a, b, m, n, k).to(torch.float32)
        torch.cuda.synchronize()
        output_torch = torch.matmul(a.to(torch.float32), b.to(torch.float32))

        if not torch.allclose(output_igemm, output_torch, rtol=1e-2):
            import IPython
            IPython.embed()

        # import IPython
        # IPython.embed()

        error_rate = compute_relative_error(output_igemm, output_torch)
        print(f"Relative Error = {error_rate:.6f}")

validation()