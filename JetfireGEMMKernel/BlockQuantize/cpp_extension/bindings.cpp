#include "include/hgemm.h"
#include "include/igemm.h"
#include <torch/extension.h>
#include <iostream>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("igemm_output_int_quantize_stochastic", &igemm_output_int_quantize_stochastic,
        "Dequantize + INT8 TensorCore GEMM + Stochastic quantized (fused into one kernel,"
        "input is INT8 and scale, output is INT8 and scale."
        "Designed for backward activation gradient");
  m.def("igemm_output_int_quantize_bias_rowcol", &igemm_output_int_quantize_bias_rowcol,
        "Dequantize + INT8 TensorCore GEMM + quantized (fused into one kernel,"
        "input is INT8 and scale, output is INT8 and scale."
        "Designed for forward process, Row major * Column Major");
  m.def("igemm_output_int_quantize_bias_rowrow", &igemm_output_int_quantize_bias_rowrow,
        "Dequantize + INT8 TensorCore GEMM + quantized (fused into one kernel,"
        "input is INT8 and scale, output is INT8 and scale."
        "Designed for forward process (Not usable)");
  m.def("igemm_output_int_quantize", &igemm_output_int_quantize,
        "Dequantize + INT8 TensorCore GEMM + quantized (fused into one kernel,"
        "input is INT8 and scale, output is INT8 and scale."
        "Designed as an template");
  m.def("igemm_output_int_quantize_fused", &igemm_output_int_quantize_fused,
        "(This function is useless) Dequantize + INT8 TensorCore GEMM + quantized (fused into GEMM), input is INT8 and scale, "
        "output is INT8 and scale."
        "Designed for backward weight gradient.");
  m.def("igemm_output_fp_no_quantize", &igemm_output_fp_no_quantize,
        "Dequantize + INT8 TensorCore GEMM, input is INT8 and scale, output is FP32/16");
  m.def("igemm_basic_int8_gemm", &igemm_basic_int8_gemm,
        "Basic INT8 TensorCore GEMM, input is INT8, output is INT32");
  m.def("hgemm_v1", &hgemm_v1,
        "Half GEMM V1");
//   m.def("quantized_divide_igemm_v1", &quantized_divide_igemm_v1,
//         "Quantized INT8 GEMM V1, quantize part is another kernel");
}

