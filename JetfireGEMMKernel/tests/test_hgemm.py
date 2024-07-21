import torch
import BlockQuantize
import BlockQuantizeCUDA as BQC

def compute_relative_error(tensor1, tensor2):
    norm_diff = torch.norm(tensor1 - tensor2)
    
    relative_error = norm_diff / torch.norm(tensor1)
    
    return relative_error.item()

dtype = torch.float16
def validation():
    for m, n, k in [(128, 256, 32)]: # 
        a = torch.rand((m, k, ), dtype=dtype, requires_grad=False).cuda() * 10
        b = torch.rand((k, n, ), dtype=dtype, requires_grad=False).cuda() * 10

        output_hgemm = BQC.hgemm_v1(a, b, m, n, k)
        torch.cuda.synchronize()
        output_torch = torch.matmul(a, b)

        if not torch.allclose(output_hgemm, output_torch, rtol=1e-2):
            import IPython
            IPython.embed()

        error_rate = compute_relative_error(output_hgemm, output_torch)
        print(f"Relative Error = {error_rate:.6f}")

validation()