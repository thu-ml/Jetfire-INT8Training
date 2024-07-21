// #include "include/igemm.h"
#include "../include/common.h"
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// Vanilla INT8 WMMA GEMM
template <typename scalar_t1, typename scalar_t2>
__global__ void igemm_basic_int8_gemm_cuda_kernel(
    scalar_t1 * __restrict__ a, scalar_t1 * __restrict__ b, scalar_t2 * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 16;
    const int BPAD = 16; // WARNING: this will cause address misalignment error

    __shared__ int8_t s_a[BM][BK + APAD];
    __shared__ int8_t s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], static_cast<int32_t>(0.0)); // Warning haocheng: this int32_t is strange 
        }
    }

    int load_a_smem_m = (tid >> 1);
    int load_a_smem_k = (tid &  1) << 4;
    int load_b_smem_k = (tid >> 4) << 1;
    int load_b_smem_n = (tid & 15) << 4;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        INT4(s_a[load_a_smem_m    ][load_a_smem_k]) = INT4(a[load_a_gmem_addr        ]);
        INT4(s_b[load_b_smem_k    ][load_b_smem_n]) = INT4(b[load_b_gmem_addr        ]);
        INT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = INT4(b[load_b_gmem_addr +     N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int32_t* ch = reinterpret_cast<int32_t*>(c);
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&ch[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

// Vanilla INT8 WMMA GEMM
torch::Tensor igemm_basic_int8_gemm_cuda(
    torch::Tensor X, torch::Tensor W,
    const int M, const int N, const int K) {
    // X shape (M, K), W shape (K, N)
    
    const int BM = 128, BN = 256;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY);

    auto option_output = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
    torch::Tensor O = torch::empty({M, N}, option_output);

    // std::cout << X.scalar_type() << std::endl;
    // std::cout << X.dtype() << std::endl;
    // std::cout << O.scalar_type() << std::endl;

    igemm_basic_int8_gemm_cuda_kernel<int8_t, int32_t><<<gridDim, blockDim>>>(
        X.data_ptr<int8_t>(),
        W.data_ptr<int8_t>(),
        O.data_ptr<int32_t>(),
        M,
        N,
        K
    );
    return O;
}
