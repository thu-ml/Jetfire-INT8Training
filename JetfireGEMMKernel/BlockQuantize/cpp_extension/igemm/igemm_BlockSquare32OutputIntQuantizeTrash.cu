// #include "include/igemm.h"
#include "../include/common.h"
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// Quantize + Dequantize Vanilla INT8 GEMM
template <typename scalar_t1, typename scalar_t2>
__global__ void igemm_output_int_quantize_fused_cuda_kernel(
    scalar_t1 * __restrict__ a, scalar_t1 * __restrict__ b, 
    half *__restrict__ sa, half *__restrict__ sb,
    int32_t * __restrict__ c, half *__restrict__ sc,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;
    
    const int QM = 32;
    const int QN = 32;
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

    __shared__ int8_t s_a[BM][BK + APAD];
    __shared__ int8_t s_b[BK][BN + BPAD];

    __shared__ half s_qa[BSM];
    __shared__ half s_qb[BSN]; // based on that BK == QK == 32

    __shared__ half s_qc[BSM][BSN];
    __shared__ half s_qc_fragment[BSM * 4][BSN * 4][32];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_fpc[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_intc[4][4];

    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_zeroc[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_zeroc[i][j], static_cast<int32_t>(0.0));
            wmma::fill_fragment(frag_fpc[i][j], static_cast<float>(0.0)); // Warning haocheng: this int32_t is strange 
            wmma::fill_fragment(frag_intc[i][j], static_cast<int32_t>(0.0)); // Warning haocheng: this int32_t is strange 
        }
    }

    // input tensor address
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

    // scale factor address
    int load_sa_gmem_m = by * BSM * NUMQK;
    int load_sb_gmem_n = bx * BSN;

    half scale_qa0, scale_qa1, scale_qb0, scale_qb1;
    float scale_qa0f, scale_qa1f, scale_qb0f, scale_qb1f;
    float scale_a, scale_b, scale_ab;

    for (int bk = 0; bk < K / BK; bk++) {
        INT4(s_a[load_a_smem_m    ][load_a_smem_k]) = INT4(a[load_a_gmem_addr        ]);
        INT4(s_b[load_b_smem_k    ][load_b_smem_n]) = INT4(b[load_b_gmem_addr        ]);
        INT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = INT4(b[load_b_gmem_addr +     N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // printf("tid: %d | %d | %d | %d \n", tid, load_a_gmem_addr, load_b_gmem_addr, a[load_a_gmem_addr]);
        // | sa: %f | sb: %f | %d | %d 
        // , a[load_a_gmem_addr], b[load_b_gmem_addr], load_a_gmem_addr, load_b_gmem_addr
        
        // per block
        // printf("START  | by: %d | bx: %d | bk: %d \n", by, bx, bk);
        // printf("wid: %d | SA: %d | SB: %d \n", wid, load_sa_gmem_m, load_sb_gmem_n);
        s_qa[0] = sa[load_sa_gmem_m        ];
        s_qa[1] = sa[load_sa_gmem_m +     NUMQK];
        s_qa[2] = sa[load_sa_gmem_m + 2 * NUMQK];
        s_qa[3] = sa[load_sa_gmem_m + 3 * NUMQK];

        s_qb[0] = sb[load_sb_gmem_n    ];
        s_qb[1] = sb[load_sb_gmem_n + 1];
        s_qb[2] = sb[load_sb_gmem_n + 2];
        s_qb[3] = sb[load_sb_gmem_n + 3];
        s_qb[4] = sb[load_sb_gmem_n + 4];
        s_qb[5] = sb[load_sb_gmem_n + 5];
        s_qb[6] = sb[load_sb_gmem_n + 6];
        s_qb[7] = sb[load_sb_gmem_n + 7];

        scale_qa0 = s_qa[2 * comp_c_frag_m    ];
        scale_qa1 = s_qa[2 * comp_c_frag_m + 1];

        // printf("%d, %d, %d | %f \n ", wid, 2 * comp_c_frag_m, bk, __half2float(scale_qa0));

        scale_qb0 = s_qb[2 * comp_c_frag_n    ];
        scale_qb1 = s_qb[2 * comp_c_frag_n + 1];


        scale_qa0f = __half2float(scale_qa0);
        scale_qa1f = __half2float(scale_qa1); 
        scale_qb0f = __half2float(scale_qb0); 
        scale_qb1f = __half2float(scale_qb1);

        // if (tid % 32 == 0){
        //     printf("tid: %d | wid: %d | %f %f %f %f \n", tid, wid, scale_qa0f, scale_qa1f, scale_qb0f, scale_qb1f);
        // }

        // FLOAT2(s_qa[0][0]) = FLOAT2(sa[load_sa_gmem_m]);
        // FLOAT4(s_qb[0][0]) = FLOAT4(sb[load_sb_gmem_n]);
    
        load_sa_gmem_m += BSK;
        load_sb_gmem_n += BSK * (N / QN);
        // printf("by: %d | bx: %d | s_qa: %f %f %f %f \n", 
        //         by, bx, 
        //         __half2float(s_qa[0][0]), __half2float(s_qa[1][0]), __half2float(s_qa[2][0]), __half2float(s_qa[3][0]));
        // printf("by: %d | bx: %d | s_qb: %f %f %f %f %f %f %f %f \n", 
        //         by, bx, 
        //         __half2float(s_qb[0][0]), __half2float(s_qb[0][1]), __half2float(s_qb[0][2]), __half2float(s_qb[0][3]),
        //         __half2float(s_qb[0][4]), __half2float(s_qb[0][5]), __half2float(s_qb[0][6]), __half2float(s_qb[0][7]));
    
        // printf("FINISH | by: %d | bx: %d | bk: %d \n", by, bx, bk);

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
            if (i == 0 || i == 1) {
                scale_a = scale_qa0f;
            } else if (i == 2 || i == 3) {
                scale_a = scale_qa1f;
            } else{
                printf("Error I = %d", i);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (j == 0 || j == 1) {
                    scale_b = scale_qb0f;
                } else if (j == 2 || j == 3) {
                    scale_b = scale_qb1f;
                } else{
                    printf("Error J = %d", j);
                }

                // scale_ab = __hmul(scale_a, scale_b);
                scale_ab = scale_a * scale_b;
                wmma::mma_sync(frag_intc[i][j], frag_a[0][i], frag_b[0][j], frag_zeroc[i][j]);
                wmma::mma_sync(frag_intc[i][j], frag_a[1][i], frag_b[1][j], frag_intc[i][j]);
                
                __syncthreads();
                // if (tid % 32 == 0){
                //     printf("bx: %d by: %d | tid: %d | wid: %d | %d %d | %f %f %f \n", bx, by, tid, wid, i, j, scale_ab, scale_a, scale_b);
                // }
                #pragma unroll
                for(int k=0; k < frag_intc[i][j].num_elements; k++) {
                    frag_fpc[i][j].x[k] = frag_fpc[i][j].x[k] + scale_ab * frag_intc[i][j].x[k];
                }
            }
        }

        __syncthreads();
    }

    // int32_t* ch = reinterpret_cast<int32_t*>(c);
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

    for (int x = 0; x < 4; x += 2) {
        for (int y = 0; y < 4; y += 2) {
            half Qscale = __double2half(0.0);
            float Qscale_inv = static_cast<float>(0.0);
            half absMaxVal = __double2half(0.0);
                                
            int store_sc_gmem_m = by * BSM + comp_c_frag_m * 2 + x / 2;
            int store_sc_gmem_n = bx * BSN + comp_c_frag_n * 2 + y / 2;
            int store_sc_gmem_addr = OFFSET(store_sc_gmem_m, store_sc_gmem_n, NUMQN);

            for (int i = x; i < x + 2; i++) {
                for (int j = y; j < y + 2; j++) {
                    wmma::fill_fragment(frag_intc[i][j], static_cast<int32_t>(0.0)); // Warning haocheng: this int32_t is strange 
                    absMaxVal = __double2half(0.0);
                    for (int k = 0; k < frag_fpc[i][j].num_elements; k++) {
                        half val = __float2half(fabs(frag_fpc[i][j].x[k]));
                        absMaxVal = (__hgt(val, absMaxVal)) ? val : absMaxVal;
                    }
                    s_qc_fragment[comp_c_frag_m * 4 + i][comp_c_frag_n * 4 + j][tid % 32] = absMaxVal;
                    __syncthreads();

                    // printf("%d, %d, %d | %d, %d | %d, %d, %d, %f, %f | %d, %d, %d | %d %d %d %d\n", 
                    //         __LINE__, wid, tid, i, j, 
                    //         by * BSM * NUMQN + bx * BSN, by * BSM, bx * BSN, 
                    //         __half2float(Qscale), __half2float(absMaxVal), 
                    //         comp_c_frag_m * 4 + i, comp_c_frag_n * 4 + j, tid % 32,
                    //         x, y, i, j);
                    // printf("%d, %d, %d | %d, %d | %d, %d, %d, %f, %f \n", __LINE__, wid, tid, i, j, by * BSM * NUMQN + bx * BSN, 
                    //         by * BSM, bx * BSN, Qscale, absMaxVal);
                    __syncthreads();
                }
            }
            
            absMaxVal = __double2half(0.0);
            for (int i = x; i < x + 2; i++) {
                for (int j = y; j < y + 2; j++) {
                    for (int t = 1; t < 32; t++) {
                        half current_value = s_qc_fragment[comp_c_frag_m * 4 + i][comp_c_frag_n * 4 + j][t];
                        if (__hgt(current_value, absMaxVal)) {
                            absMaxVal = current_value;
                        }
                    }
                }
            }
            
            Qscale = __float2half(__half2float(absMaxVal) / static_cast<float>(127));
            Qscale_inv = static_cast<float>(127) / __half2float(absMaxVal);
            sc[store_sc_gmem_addr] = Qscale;

            // printf("%d, %d, %d | %d, %d, %d, %f, %f | %d %d | %d %d %d %d %d \n", 
            //         __LINE__, wid, tid, 
            //         by * BSM * NUMQN + bx * BSN, by * BSM, bx * BSN, 
            //         __half2float(Qscale), __half2float(absMaxVal), 
            //         tid % 32, store_sc_gmem_addr,
            //         x, y, store_c_gmem_m, store_c_gmem_n, NUMQN);

            for (int i = x; i < x + 2; i++) {
                for (int j = y; j < y + 2; j++) {
                    // printf("%d, %d, %d | %d, %d | %d, %d, %d, %f, %f \n", __LINE__, wid, tid, i, j, by * BSM * NUMQN + bx * BSN, 
                    //         by * BSM, bx * BSN, Qscale, absMaxVal);

                    #pragma unroll
                    for(int k=0; k < frag_fpc[i][j].num_elements; k++) {
                        frag_intc[i][j].x[k] = std::clamp((int)(Qscale_inv * frag_fpc[i][j].x[k]), -127, 127);
                    }
                    
                    wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_intc[i][j], N, wmma::mem_row_major);
                }
            }
        }
    }


    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {

    //     }
    // }
}

// Quantize + Dequantize Vanilla INT8 GEMM
std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_fused_cuda(
    torch::Tensor X, torch::Tensor W,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K) {
    // X shape (M, K), W shape (K, N)
    
    const int BM = 128, BN = 256;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY);

    int QM = SX.size(1);
    int QK = SX.size(0);
    int QN = SW.size(1);

    auto option_output = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
    torch::Tensor O = torch::empty({M, N}, option_output);

    auto option_scale_output = torch::TensorOptions().dtype(torch::kFloat16).device(X.device());
    torch::Tensor SO = torch::empty({QM, QN}, option_scale_output);

    // std::cout << X.scalar_type() << std::endl;
    // std::cout << X.dtype() << std::endl;
    // std::cout << O.scalar_type() << std::endl;

    // std::cout << SX << std::endl;
    // std::cout << SW << std::endl;

    igemm_output_int_quantize_fused_cuda_kernel<int8_t, int32_t><<<gridDim, blockDim>>>(
        X.data_ptr<int8_t>(),
        W.data_ptr<int8_t>(),
        (half*)SX.data_ptr<at::Half>(),
        (half*)SW.data_ptr<at::Half>(),
        O.data_ptr<int32_t>(),
        (half*)SO.data_ptr<at::Half>(),
        M,
        N,
        K
    );
    return std::make_tuple(O, SO);
}