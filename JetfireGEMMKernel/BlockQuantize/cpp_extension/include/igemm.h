#include <torch/types.h>

torch::Tensor igemm_basic_int8_gemm_cuda(
    torch::Tensor X, torch::Tensor W,
    const int M, const int N, const int K);

torch::Tensor igemm_basic_int8_gemm(
    torch::Tensor X, torch::Tensor W,
    const int M, const int N, const int K){

    return igemm_basic_int8_gemm_cuda(X, W, M, N, K);
}

torch::Tensor igemm_output_fp_no_quantize_cuda(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K);

torch::Tensor igemm_output_fp_no_quantize(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K){

    return igemm_output_fp_no_quantize_cuda(X, W, SX, SW, M, N, K);
} 

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_fused_cuda(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K);

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_fused(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K){

    return igemm_output_int_quantize_fused_cuda(X, W, SX, SW, M, N, K);
} 

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_cuda(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K);

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K){

    return igemm_output_int_quantize_cuda(X, W, SX, SW, M, N, K);
} 

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_bias_rowrow_cuda(
    torch::Tensor X, torch::Tensor W, torch::Tensor bias, torch::Tensor biasmax,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K);

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_bias_rowrow(
    torch::Tensor X, torch::Tensor W, torch::Tensor bias, torch::Tensor biasmax,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K){

    return igemm_output_int_quantize_bias_rowrow_cuda(X, W, bias, biasmax, SX, SW, M, N, K);
} 

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_bias_rowcol_cuda(
    torch::Tensor X, torch::Tensor W, torch::Tensor bias, torch::Tensor biasmax,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K);

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_bias_rowcol(
    torch::Tensor X, torch::Tensor W, torch::Tensor bias, torch::Tensor biasmax,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K){

    return igemm_output_int_quantize_bias_rowcol_cuda(X, W, bias, biasmax, SX, SW, M, N, K);
} 

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_stochastic_cuda(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K);

std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_stochastic(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K){

    return igemm_output_int_quantize_stochastic_cuda(X, W, SX, SW, M, N, K);
} 
// std::tuple<torch::Tensor, torch::Tensor> quantized_igemm_v1_cuda(
//     torch::Tensor X, torch::Tensor W,
//     torch::Tensor SX, torch::Tensor SW,
//     const int M, const int N, const int K);

// std::tuple<torch::Tensor, torch::Tensor> quantized_igemm_v1(
//     torch::Tensor X, torch::Tensor W,
//     torch::Tensor SX, torch::Tensor SW,
//     const int M, const int N, const int K){

//     return quantized_igemm_v1_cuda(X, W, SX, SW, M, N, K);
// } 