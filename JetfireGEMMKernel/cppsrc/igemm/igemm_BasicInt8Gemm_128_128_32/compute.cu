// #include "include/igemm.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <curand.h>
#include <float.h>
#include <curand_kernel.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void cpuI8I32Gemm(int8_t *a, int8_t *b, int32_t *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (int32_t)psum;
        }
    }
}

// Vanilla INT8 WMMA GEMM
template <typename scalar_t1, typename scalar_t2>
__global__ void igemm_basic_int8_gemm_cuda_kernel(
    scalar_t1 * __restrict__ a, scalar_t1 * __restrict__ b, scalar_t2 * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 16;
    const int BPAD = 16; // WARNING: this will cause address misalignment error

    __shared__ int8_t s_a[BM][BK + APAD];
    __shared__ int8_t s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[2][4]; // 64 * 32
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> frag_b[2][4]; // 32 * 64
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], static_cast<int32_t>(0.0)); // Warning haocheng: this int32_t is strange 
        }
    }

    // 128 thread, one load 32 element 
    int load_a_smem_m = (tid >> 1) << 1;
    int load_a_smem_k = (tid &  1) << 4;
    int load_b_smem_k = (tid >> 3) << 1;
    int load_b_smem_n = (tid &  7) << 4;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        INT4(s_a[load_a_smem_m    ][load_a_smem_k]) = INT4(a[load_a_gmem_addr        ]);
        INT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = INT4(a[load_a_gmem_addr     + K]);
        INT4(s_b[load_b_smem_k    ][load_b_smem_n]) = INT4(b[load_b_gmem_addr        ]);
        INT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = INT4(b[load_b_gmem_addr     + N]);

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

typedef enum{
    igemm_kernel
} IGemmTCAlgo_t;

template<IGemmTCAlgo_t algo = igemm_kernel>
void myI8I32GemmTCWarp(int8_t *a, int8_t *b, int32_t *c, int M, int N, int K) {

    if (algo == igemm_kernel) {
        const int BM = 128, BN = 128;
        dim3 blockDim(128);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;
        dim3 gridDim(BX, BY);
        igemm_basic_int8_gemm_cuda_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    }
}

template <typename scalar_t1>
void printArrayByRowsToFile(const scalar_t1* arr, int M, int N, const std::string& filename) {
    std::ofstream outputFile(filename);

    if (outputFile.is_open()) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int index = i * N + j;
                outputFile << std::fixed << std::setprecision(6) << static_cast<int32_t>(arr[index]) << " ";
            }
            outputFile << std::endl;
        }

        outputFile.close();
    } else {
        std::cout << "Unable to open the file." << std::endl;
    }
}

float testI8I32GemmMaxError(
    void (*gpuI8I32Gemm) (int8_t *, int8_t *, int32_t *, int, int, int),
    int M, int N, int K) {

    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_int_distribution<int8_t> distribution(-127, 127);
    
    size_t size_a = M * K * sizeof(int8_t);
    size_t size_b = K * N * sizeof(int8_t);
    size_t size_c = M * N * sizeof(int32_t);

    int8_t *h_a, *h_b, *d_a, *d_b;
    int32_t *h_c, *d_c, *h_d_c;
    h_a = (int8_t *)malloc(size_a);
    h_b = (int8_t *)malloc(size_b);
    h_c = (int32_t *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (int32_t *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = distribution(generator);
    for (int i = 0; i < K * N; i++)
        h_b[i] = distribution(generator);
    
    
    cpuI8I32Gemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuI8I32Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

    return max_error;
}

__global__ void generateInt8Data(int8_t* data, int size, unsigned int seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    curandState_t state;
    curand_init(seed, index, 0, &state);

    if (index < size) {
        int randValue = curand(&state) % 255 - 127;  // 生成-127到127之间的随机数
        data[index] = static_cast<int8_t>(randValue);
    }
}

float testI8I32GemmPerformance(
    void (*gpuI8I32Gemm) (int8_t *, int8_t *, int32_t *, int, int, int),
    int M, int N, int K, int repeat) {

    size_t size_a = M * K * sizeof(int8_t);
    size_t size_b = K * N * sizeof(int8_t);
    size_t size_c = M * N * sizeof(int32_t);
    int num_a = M * K;
    int num_b = K * N;

    int8_t *d_a, *d_b;
    int32_t *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    dim3 blockDim(128);
    dim3 gridDima((num_a + blockDim.x - 1) / blockDim.x);
    dim3 gridDimb((num_b + blockDim.x - 1) / blockDim.x);

    generateInt8Data<<<gridDima, blockDim>>>(d_a, num_a, 0);
    generateInt8Data<<<gridDimb, blockDim>>>(d_b, num_b, 0);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        gpuI8I32Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}

int main(){
    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];
    for (int i = 0; i < test_num; i++) {
        M_list[i] = (i + 1) * 256;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    const int outer_repeat = 10, inner_repeat = 1;

    {
        printf("\nalgo = HGEMMAlignedV1\n");

        {
            const int M = 128, N = 128, K = 32;
            float max_error = testI8I32GemmMaxError(
                myI8I32GemmTCWarp<igemm_kernel>, M, N, K);
            printf("Max Error when M = %d, N = %d, K = %d is %f\n", M, N, K, max_error);
        }

        {
            const int M = 512, N = 512, K = 128;
            float max_error = testI8I32GemmMaxError(
                myI8I32GemmTCWarp<igemm_kernel>, M, N, K);
            printf("Max Error when M = %d, N = %d, K = %d is %f\n", M, N, K, max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testI8I32GemmPerformance(
                    myI8I32GemmTCWarp<igemm_kernel>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Tflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
        }
    }

    return 0;
}