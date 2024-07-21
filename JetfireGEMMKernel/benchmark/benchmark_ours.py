import os
import time
import torch
import numpy as np
from tqdm import tqdm

import json

import MyGEMM

test_num = 64
repeat_time = 50
M_list, N_list, K_list = np.arange(1, test_num) * 256, np.arange(1, test_num) * 256, np.arange(1, test_num) * 256

def calculate_inference_speed_time(method):
    # self time
    avg_self_secs = []
    avg_self_tflops = []
    for m, n, k in tqdm(zip(M_list, N_list, K_list), total=test_num):
        m, n, k = 8192, 8192, 8192
        # Regular Tensor
        a32 = torch.rand((m, k, ), dtype=torch.float32, requires_grad=False).cuda() * 10
        b32 = torch.rand((k, n, ), dtype=torch.float32, requires_grad=False).cuda() * 10
        bias = torch.rand((n,), dtype=torch.float32, requires_grad=False).cuda()

        a16, b16 = a32.to(torch.float16), b32.to(torch.float16)

        a8, b8 = a16.to(torch.int8), b16.to(torch.int8)

        # Quantized Tensor
        b = 32
        x = torch.rand((m, k, ), dtype=torch.float16, requires_grad=False).cuda() * 100
        w = torch.rand((k, n, ), dtype=torch.float16, requires_grad=False).cuda() * 100
        x = (x - x.mean()) / np.sqrt(x.numel())
        w = (w - w.mean()) / np.sqrt(w.numel())

        # triton result
        _qx = x.reshape(m // b, b, k // b, b).permute(0, 2, 1, 3)
        _qw = w.reshape(k // b, b, n // b, b).permute(0, 2, 1, 3)
        sx = _qx.abs().amax(dim=(2, 3)) / 127
        sw = _qw.abs().amax(dim=(2, 3)) / 127
        
        _qx = ((_qx / sx.unsqueeze(2).unsqueeze(3)).round()).to(torch.int8)
        _qw = ((_qw / sw.unsqueeze(2).unsqueeze(3)).round()).to(torch.int8)

        qx = _qx.permute(0, 2, 1, 3).reshape(m, k)
        qw = _qw.permute(0, 2, 1, 3).reshape(k, n)

        torch.cuda.synchronize()
        
        # actual speed
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat_time)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat_time)]
        for i in range(repeat_time):
            start_event[i].record()
            if method == "HV1":
                from BlockQuantize._CUDA import hgemm_v1
                self_output = hgemm_v1(a16, b16, m, n, k)
            elif method == "HV2":
                raise NotImplementedError("HV2 not implemented")
            elif method == "HV3":
                raise NotImplementedError("HV3 not implemented")
            elif method == "HV4":
                raise NotImplementedError("HV4 not implemented")
            elif method == "HV5":
                raise NotImplementedError("HV5 not implemented")
            elif method == "torch-8":
                raise ValueError("INT8 matmul not supported by torch")
            elif method == "torch-16":
                self_output = torch.matmul(a16, b16)
            elif method == "torch-32":
                self_output = torch.matmul(a32, b32)
            elif method == "BasicInt8":
                # from BlockQuantize._CUDA import quantized_igemm_v1
                import BlockQuantizeCUDA as BQC
                self_output = BQC.igemm_basic_int8_gemm(qx, qw, m, n, k)
            elif method == "DQInt8Fp":
                import BlockQuantizeCUDA as BQC
                self_output = BQC.igemm_output_fp_no_quantize(qx, qw, sx, sw, m, n, k)
            elif method == "DQInt8QTrash":
                import BlockQuantizeCUDA as BQC
                self_output = BQC.igemm_output_int_quantize_fused(qx, qw, sx, sw, m, n, k)
            elif method == "DQInt8Q":
                import BlockQuantizeCUDA as BQC
                self_output = BQC.igemm_output_int_quantize(qx, qw, sx, sw, m, n, k)
            elif method == "DQInt8QStochastic":
                import BlockQuantizeCUDA as BQC
                self_output = BQC.igemm_output_int_quantize_stochastic(qx, qw, sx, sw, m, n, k)
            elif method == "DQInt8QBiasRowRow":
                import BlockQuantizeCUDA as BQC
                biasmax = bias.max()
                self_output = BQC.igemm_output_int_quantize_bias_rowrow(qx, qw, bias, biasmax, sx, sw, m, n, k)
            elif method == "DQInt8QBiasRowCol":
                import BlockQuantizeCUDA as BQC
                biasmax = bias.max()
                self_output = BQC.igemm_output_int_quantize_bias_rowcol(qx, qw, bias, biasmax, sx, sw, m, n, k)
            else:
                raise ValueError("Not supported benchmark type")

            end_event[i].record()
        torch.cuda.synchronize()
        
        total_self_sec = 1e-3 * torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        avg_self_sec = torch.quantile(total_self_sec, 0.5)
        avg_self_tflop = 2 * m * n * k / avg_self_sec / np.power(1024, 4)

        print(f"M: {m}, Tflops: {avg_self_tflop}")
        print(f"M: {m}, Seconds: {avg_self_sec}")

        avg_self_secs.append(avg_self_sec)
        avg_self_tflops.append(avg_self_tflop)

    os.makedirs(f"result/{method}", exist_ok=True)
    result_dict = {}
    result_dict["M_list"] = M_list
    result_dict[f"avg_{method}_secs"] = avg_self_secs
    result_dict[f"avg_{method}_tflops"] = avg_self_tflops
    np.save(f"result/{method}/result.npy", result_dict)

    # print("self", m, n, k, avg_self_sec, avg_self_tflop)

for method in ["torch-16", "torch-32", "DQInt8QBiasRowRow", "DQInt8QBiasRowCol", "DQInt8QStochastic", "DQInt8Fp", "DQInt8Q"]:
# for method in ["INT8V1", "torch-16", "torch-32", "HV1"]:
    calculate_inference_speed_time(method)

import IPython
IPython.embed()