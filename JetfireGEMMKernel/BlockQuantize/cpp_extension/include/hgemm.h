#include <torch/types.h>

torch::Tensor hgemm_v1_cuda(
    torch::Tensor X, torch::Tensor W,
    const int M, const int N, const int K);

torch::Tensor hgemm_v1(
    torch::Tensor X, torch::Tensor W,
    const int M, const int N, const int K){

    return hgemm_v1_cuda(X, W, M, N, K);
}