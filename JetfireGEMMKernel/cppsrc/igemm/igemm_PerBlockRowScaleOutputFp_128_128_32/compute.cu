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
#define FLOAT1(pointer) (reinterpret_cast<float1*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void cpuI8F32Gemm(int8_t *a, int8_t *b, half *sa, half *sb, float *c, int M, int N, int K, int QM, int QN, int QK) {
    
    const int NUMQM = M / QM;
    const int NUMQK = K / QK;
    const int NUMQN = N / QN;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                int sm = m / QM;
                int sn = n / QN;
                int sk = k / QK;
                // printf("%f, %f, %f, %f \n", static_cast<float>(a[OFFSET(m, k, K)]), static_cast<float>(b[OFFSET(k, n, N)]), static_cast<float>(sa[OFFSET(sm, sk, NUMQK)]), static_cast<float>(sb[OFFSET(sk, sn, NUMQN)]));
                psum += static_cast<float>(a[OFFSET(m, k, K)]) * static_cast<float>(b[OFFSET(k, n, N)]) * static_cast<float>(sa[OFFSET(sm, sk, NUMQK)]) * static_cast<float>(sb[OFFSET(sk, sn, NUMQN)]);
            }
            c[OFFSET(m, n, N)] = static_cast<float>(psum);
            // printf("%d %d %f \n", m, n, c[OFFSET(m, n, N)]);
        }
    }
}

// Quantize + Dequantize Vanilla INT8 GEMM
template <typename scalar_t1, typename scalar_t2>
__global__ void igemm_output_fp_no_quantize_blockperrow_cuda_kernel(
    scalar_t1 * __restrict__ a, scalar_t1 * __restrict__ b, 
    half *__restrict__ sa, half *__restrict__ sb,
    float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 32;
    
    const int QM = 1;
    const int QN = 1;
    const int QK = 32;

    const int BSM = BM / QM;
    const int BSK = BK / QK;
    const int BSN = BN / QN;

    const int NUMQM = M / QM;
    const int NUMQK = K / QK;
    const int NUMQN = N / QN;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 16;
    const int BPAD = 16; // WARNING: this will cause address misalignment error
    const int SPAD = 8; // WARNING: this will cause address misalignment error

    __shared__ int8_t s_a[BM][BK + APAD];
    __shared__ int8_t s_b[BK][BN + BPAD];

    __shared__ half s_scale[BM][BN + SPAD];

    __shared__ half s_qa[128]; // one thread load 4 element (FLOAT load)
    __shared__ half s_qb[128]; // one thread load 8 element (FLOAT2 load)

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_fpc[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_intc[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_zero[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_scale[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_scale[i][j], static_cast<half>(0.0));
            wmma::fill_fragment(frag_fpc[i][j], static_cast<float>(0.0)); // Warning haocheng: this int32_t is strange 
            wmma::fill_fragment(frag_intc[i][j], static_cast<int32_t>(0.0)); // Warning haocheng: this int32_t is strange 
            wmma::fill_fragment(frag_zero[i][j], static_cast<int32_t>(0.0)); // Warning haocheng: this int32_t is strange 
        }
    }

    // 128 thread, one load 32 element 
    int load_a_smem_m = (tid >> 1) << 1;
    int load_a_smem_k = (tid &  1) << 4;
    int load_b_smem_k = (tid >> 3) << 1;
    int load_b_smem_n = (tid &  7) << 4;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K); // load 16 element per thread, use INT4 ()
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N); // load 32 element per thread, use 2 * INT4

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    // scale factor address
    int load_sa_smem_m = tid;
    int load_sb_smem_n = tid;

    int load_sa_gmem_m = by * BSM + load_sa_smem_m;
    int load_sb_gmem_n = bx * BSN + load_sb_smem_n;

    int load_sa_gmem_addr = OFFSET(load_sa_gmem_m, 0, NUMQK); // load 16 element per thread, use INT4 ()
    int load_sb_gmem_addr = OFFSET(0, load_sb_gmem_n, NUMQN); // load 32 element per thread, use 2 * INT4

    for (int bk = 0; bk < K / BK; bk++) {
        INT4(s_a[load_a_smem_m    ][load_a_smem_k]) = INT4(a[load_a_gmem_addr        ]);
        INT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = INT4(a[load_a_gmem_addr     + K]);
        INT4(s_b[load_b_smem_k    ][load_b_smem_n]) = INT4(b[load_b_gmem_addr        ]);
        INT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = INT4(b[load_b_gmem_addr     + N]);

        // printf("%d: %f %f %d %d \n", tid, __half2float(s_qa[load_sa_smem_m]), __half2float(s_qb[load_sb_smem_n]), load_sa_smem_m, load_sb_smem_n);
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        s_qa[load_sa_smem_m] = sa[load_sa_gmem_addr];
        s_qb[load_sb_smem_n] = sb[load_sb_gmem_addr];

        load_sa_gmem_addr += BSK;
        load_sb_gmem_addr += BSK * NUMQN;

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BN; i++){
            s_scale[tid][i] = __hmul(s_qa[tid], s_qb[i]); // OFFSET(tid, i, BN)
            // if (__half2float(s_scale[tid][i]) == INFINITY) {printf("%d: %f %f %f %d %d \n", i, __half2float(s_scale[tid][i]), __half2float(s_qa[tid]), __half2float(s_qb[i]), tid, i);}
        }

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

        wmma::load_matrix_sync(frag_scale[0][0], &s_scale[comp_c_frag_m * 64     ][comp_c_frag_n * 64     ], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[1][0], &s_scale[comp_c_frag_m * 64 + 16][comp_c_frag_n * 64     ], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[2][0], &s_scale[comp_c_frag_m * 64 + 32][comp_c_frag_n * 64     ], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[3][0], &s_scale[comp_c_frag_m * 64 + 48][comp_c_frag_n * 64     ], BN + SPAD, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_scale[0][1], &s_scale[comp_c_frag_m * 64     ][comp_c_frag_n * 64 + 16], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[1][1], &s_scale[comp_c_frag_m * 64 + 16][comp_c_frag_n * 64 + 16], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[2][1], &s_scale[comp_c_frag_m * 64 + 32][comp_c_frag_n * 64 + 16], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[3][1], &s_scale[comp_c_frag_m * 64 + 48][comp_c_frag_n * 64 + 16], BN + SPAD, wmma::mem_row_major);
        
        wmma::load_matrix_sync(frag_scale[0][2], &s_scale[comp_c_frag_m * 64     ][comp_c_frag_n * 64 + 32], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[1][2], &s_scale[comp_c_frag_m * 64 + 16][comp_c_frag_n * 64 + 32], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[2][2], &s_scale[comp_c_frag_m * 64 + 32][comp_c_frag_n * 64 + 32], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[3][2], &s_scale[comp_c_frag_m * 64 + 48][comp_c_frag_n * 64 + 32], BN + SPAD, wmma::mem_row_major);
        
        wmma::load_matrix_sync(frag_scale[0][3], &s_scale[comp_c_frag_m * 64     ][comp_c_frag_n * 64 + 48], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[1][3], &s_scale[comp_c_frag_m * 64 + 16][comp_c_frag_n * 64 + 48], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[2][3], &s_scale[comp_c_frag_m * 64 + 32][comp_c_frag_n * 64 + 48], BN + SPAD, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_scale[3][3], &s_scale[comp_c_frag_m * 64 + 48][comp_c_frag_n * 64 + 48], BN + SPAD, wmma::mem_row_major);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_intc[i][j], frag_a[0][i], frag_b[0][j], frag_zero[i][j]);
                wmma::mma_sync(frag_intc[i][j], frag_a[1][i], frag_b[1][j], frag_intc[i][j]);
                #pragma unroll
                for(int t = 0; t < frag_intc[i][j].num_elements; t++){
                    frag_fpc[i][j].x[t] += (static_cast<float>(frag_intc[i][j].x[t]) * __half2float(frag_scale[i][j].x[t])); // __half2float(__hmul(__int2half_rn(frag_intc[i][j].x[t]), frag_scale[i][j].x[t]));
                    // if (frag_fpc[i][j].x[t] == INFINITY){
                    //     printf("%d %d %d %d %d | %f %f \n", tid, wid, i, j, t, static_cast<float>(frag_intc[i][j].x[t]), __half2float(frag_scale[i][j].x[t]));
                    // }
                }

            }
        }

        __syncthreads();
    }

    // int32_t* ch = reinterpret_cast<int32_t*>(c);
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_fpc[i][j], N, wmma::mem_row_major);
        }
    }
}

typedef enum{
    igemm_kernel
} IGemmTCAlgo_t;

template<IGemmTCAlgo_t algo = igemm_kernel>
void myI8F32GemmTCWarp(int8_t *a, int8_t *b, half *sa, half *sb, float *c, int M, int N, int K) {

    if (algo == igemm_kernel) {
        const int BM = 128, BN = 128;
        dim3 blockDim(128);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;
        dim3 gridDim(BX, BY);
        igemm_output_fp_no_quantize_blockperrow_cuda_kernel<int8_t, int32_t><<<gridDim, blockDim>>>(a, b, sa, sb, c, M, N, K);
    }
}

template <typename scalat_t>
void printArrayByRowsToFile(const scalat_t* arr, int M, int N, const std::string& filename) {
    FILE* outputFile = fopen(filename.c_str(), "w");

    if (outputFile != nullptr) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int index = i * N + j;
                fprintf(outputFile, "%.6f ", static_cast<float>(arr[index]));
            }
            fprintf(outputFile, "\n");
        }
        fclose(outputFile);
    } else {
        std::cout << "Unable to open the file." << std::endl;
    }
}

float testI8F32GemmMaxError(
    void (*gpuI8F32Gemm) (int8_t *, int8_t *, half *sa, half *sb, float *, int, int, int),
    int M, int N, int K, int QM, int QN, int QK) {

    // std::random_device rd;
    std::mt19937 generator(1); // rd()

    std::uniform_int_distribution<int8_t> distribution(-127, 127);

    const int NUMQM = M / QM;
    const int NUMQK = K / QK;
    const int NUMQN = N / QN;

    size_t size_a = M * K * sizeof(int8_t);
    size_t size_b = K * N * sizeof(int8_t);
    size_t size_sa = NUMQM * NUMQK * sizeof(half);
    size_t size_sb = NUMQK * NUMQN * sizeof(half);
    size_t size_c = M * N * sizeof(float);

    int8_t *h_a, *h_b, *d_a, *d_b;
    half *h_sa, *h_sb, *d_sa, *d_sb;
    float *h_c, *d_c, *h_d_c;
    h_a = (int8_t *)malloc(size_a);
    h_b = (int8_t *)malloc(size_b);
    h_sa = (half *)malloc(size_sa);
    h_sb = (half *)malloc(size_sb);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_sa, size_sa);
    cudaMalloc(&d_sb, size_sb);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++){
        h_a[i] = distribution(generator);
    }
    for (int i = 0; i < K * N; i++){
        h_b[i] = distribution(generator);
    }
    for (int i = 0; i < NUMQM * NUMQK; i++){
        h_sa[i] = float(distribution(generator)) / 127;
        // if (i == 16) printf("h_sa[%d]: %f \n", i, __half2float(h_sa[i]));
    }
    for (int i = 0; i < NUMQK * NUMQN; i++){
        h_sb[i] = float(distribution(generator)) / 127;
        // if (i < 3) printf("h_sb[%d]: %f \n", i, __half2float(h_sb[i]));
    }    
    cpuI8F32Gemm(h_a, h_b, h_sa, h_sb, h_c, M, N, K, QM, QN, QK);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sa, h_sa, size_sa, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb, h_sb, size_sb, cudaMemcpyHostToDevice);
    gpuI8F32Gemm(d_a, d_b, d_sa, d_sb, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    float max_relative_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error){ // nan
            max_error = -NAN;
        } else {
            max_error = max(max_error, this_error);
        }
    }

    std::string AFileName = "A.txt";
    std::string BFileName = "B.txt";
    printArrayByRowsToFile<int8_t>(h_a, M, K, AFileName);
    printArrayByRowsToFile<int8_t>(h_b, K, N, BFileName);
    std::string CpuFileName = "CPU.txt";
    std::string GpuFileName = "GPU.txt";
    printArrayByRowsToFile<float>(h_c, M, N, CpuFileName);
    printArrayByRowsToFile<float>(h_d_c, M, N, GpuFileName);

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

__global__ void generateHalfData(half* data, int size, unsigned int seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    curandState_t state;
    curand_init(seed, index, 0, &state);

    if (index < size) {
        float randValue = float(curand(&state)) / float(INT32_MAX);  // 生成-127到127之间的随机数
        data[index] = static_cast<half>(randValue);
    }
}

float testI8F32GemmPerformance(
    void (*gpuI8F32Gemm) (int8_t *, int8_t *, half *sa, half *sb, float *, int, int, int),
    int M, int N, int K, int QM, int QN, int QK, int repeat) {

    const int NUMQM = M / QM;
    const int NUMQK = K / QK;
    const int NUMQN = N / QN;

    size_t size_a = M * K * sizeof(int8_t);
    size_t size_b = K * N * sizeof(int8_t);
    size_t size_sa = NUMQM * NUMQK * sizeof(half);
    size_t size_sb = NUMQK * NUMQN * sizeof(half);
    size_t size_c = M * N * sizeof(float);
    int num_a = M * K;
    int num_b = K * N;
    int num_sa = NUMQM * NUMQK;
    int num_sb = NUMQK * NUMQN;

    int8_t *d_a, *d_b;
    half *d_sa, *d_sb;
    float *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_sa, size_sa);
    cudaMalloc(&d_sb, size_sb);
    cudaMalloc(&d_c, size_c);

    dim3 blockDim(128);
    dim3 gridDima((num_a + blockDim.x - 1) / blockDim.x);
    dim3 gridDimb((num_b + blockDim.x - 1) / blockDim.x);
    dim3 gridDimsa((num_sa + blockDim.x - 1) / blockDim.x);
    dim3 gridDimsb((num_sb + blockDim.x - 1) / blockDim.x);

    generateInt8Data<<<gridDima, blockDim>>>(d_a, num_a, 0);
    generateInt8Data<<<gridDimb, blockDim>>>(d_b, num_b, 0);
    generateHalfData<<<gridDimsa, blockDim>>>(d_sa, num_sa, 0);
    generateHalfData<<<gridDimsb, blockDim>>>(d_sb, num_sb, 0);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        gpuI8F32Gemm(d_a, d_b, d_sa, d_sb, d_c, M, N, K);
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
    const int QM = 1, QN = 1, QK = 32;

    const int outer_repeat = 10, inner_repeat = 1;

    {
        printf("\nalgo = HGEMMAlignedV1\n");

        {
            const int M = 128, N = 128, K = 32;
            float max_error = testI8F32GemmMaxError(
                myI8F32GemmTCWarp<igemm_kernel>, M, N, K, QM, QN, QK);
            printf("Max Error when M = %d, N = %d, K = %d is %f \n", M, N, K, max_error);
        }
        {
            const int M = 512, N = 512, K = 128;
            float max_error = testI8F32GemmMaxError(
                myI8F32GemmTCWarp<igemm_kernel>, M, N, K, QM, QN, QK);
            printf("Max Error when M = %d, N = %d, K = %d is %f \n", M, N, K, max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testI8F32GemmPerformance(
                    myI8F32GemmTCWarp<igemm_kernel>, M, N, K, QM, QN, QK, inner_repeat);
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