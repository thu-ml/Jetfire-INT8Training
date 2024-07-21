// #include "include/igemm.h"
#include "../include/common.h"
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define INT1(pointer) (reinterpret_cast<int1*>(&(pointer))[0])
#define INT2(pointer) (reinterpret_cast<int2*>(&(pointer))[0])
#define INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__device__ int8_t clamprowcol(int value, int8_t minVal, int8_t maxVal) {
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : static_cast<int8_t>(value);
}

// Quantize + Dequantize Vanilla INT8 GEMM
template <typename scalar_t1, typename scalar_t2>
__global__ void igemm_output_int_quantize_bias_rowcol_cuda_kernel(
    scalar_t1 * __restrict__ a, scalar_t1 * __restrict__ b, 
    float * __restrict__ bias, float * __restrict__ biasmax,
    half *__restrict__ sa, half *__restrict__ sb,
    int8_t * __restrict__ c, half *__restrict__ sc,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
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
    int tid_mod = tid % 32;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 16;
    const int BPAD = 16; // WARNING: this will cause address misalignment error
    const int ACCPAD = 4;

    extern __shared__ half sharedMem[];
    int8_t (*s_a)[BK + APAD] = reinterpret_cast<int8_t (*)[BK + APAD]>(sharedMem);
    int8_t (*s_b)[BK + BPAD] = reinterpret_cast<int8_t (*)[BK + BPAD]>(&s_a[2 * BM]);
    // __shared__ float thread_max[2][64];

    float (*acc_float)[2 * QN + ACCPAD] = reinterpret_cast<float (*)[2 * QN + ACCPAD]>(&sharedMem);
    float (*thread_max)[64] = reinterpret_cast<float (*)[64]>(&acc_float[2 * QM]);

    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BN * (BK + BPAD);

    // __shared__ int8_t s_a[BM][BK + APAD];
    // __shared__ int8_t s_b[BK][BN + BPAD];

    half s_qa[BSM];
    half s_qb[BSN]; // based on that BK == QK == 32

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_fpc[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_intc[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_fpc[i][j], static_cast<float>(0.0)); // Warning haocheng: this int32_t is strange 
            wmma::fill_fragment(frag_intc[i][j], static_cast<int32_t>(0.0)); // Warning haocheng: this int32_t is strange 
        }
    }

    // input tensor address
    int load_a_smem_m = (tid >> 1) << 1;
    int load_a_smem_k = (tid &  1) << 4;
    int load_b_smem_n = (tid >> 1) << 1;
    int load_b_smem_k = (tid &  1) << 4;

    int s_a_base_addr = __cvta_generic_to_shared(s_a[0]);
    int s_b_base_addr = __cvta_generic_to_shared(s_b[0]);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(int8_t);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(int8_t);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, BK + BPAD) * sizeof(int8_t);
    int load_b_smem_addr_1 = load_b_smem_addr_0 + (BK + BPAD) * sizeof(int8_t);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_smem_k, K);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    // scale factor address
    int load_sa_gmem_m = by * BSM;
    int load_sb_gmem_n = bx * BSN;

    half scale_qa0, scale_qa1, scale_qb0, scale_qb1;
    float scale_qa0f, scale_qa1f, scale_qb0f, scale_qb1f;
    float scale_a, scale_b, scale_ab;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     K]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    #pragma unroll 64
    for (int bk = 1; bk < K / BK; bk++) {
        // INT4(s_a[load_a_smem_m    ][load_a_smem_k]) = INT4(a[load_a_gmem_addr        ]);
        // INT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = INT4(a[load_a_gmem_addr     + K]);
        // INT4(s_b[load_b_smem_k    ][load_b_smem_n]) = INT4(b[load_b_gmem_addr        ]);
        // INT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = INT4(b[load_b_gmem_addr     + N]);

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(int8_t)), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(int8_t)), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(int8_t)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(int8_t)), "l"(&b[load_b_gmem_addr +     K]));

        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * BM + comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * BM + comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * BN + comp_c_frag_n * 64     ][ 0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 16][ 0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 32][ 0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 48][ 0], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * BN + comp_c_frag_n * 64     ][16], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 16][16], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 32][16], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 48][16], BK + BPAD);

        FLOAT2(s_qa[0]) = FLOAT2(sa[load_sa_gmem_m        ]);
        FLOAT2(s_qb[0]) = FLOAT2(sb[load_sb_gmem_n    ]);

        scale_qa0 = s_qa[2 * comp_c_frag_m    ];
        scale_qa1 = s_qa[2 * comp_c_frag_m + 1];

        scale_qb0 = s_qb[2 * comp_c_frag_n    ];
        scale_qb1 = s_qb[2 * comp_c_frag_n + 1];

        scale_qa0f = __half2float(scale_qa0);
        scale_qa1f = __half2float(scale_qa1); 
        scale_qb0f = __half2float(scale_qb0); 
        scale_qb1f = __half2float(scale_qb1);

        load_sa_gmem_m += BSK * NUMQM;
        load_sb_gmem_n += BSK * NUMQN;

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

                scale_ab = scale_a * scale_b;
                wmma::fill_fragment(frag_intc[i][j], static_cast<int32_t>(0.0));
                wmma::mma_sync(frag_intc[i][j], frag_a[0][i], frag_b[0][j], frag_intc[i][j]);
                wmma::mma_sync(frag_intc[i][j], frag_a[1][i], frag_b[1][j], frag_intc[i][j]);

                #pragma unroll
                for(int k=0; k < frag_intc[i][j].num_elements; k++) {
                    frag_fpc[i][j].x[k] += scale_ab * frag_intc[i][j].x[k];
                }
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;
    wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * BM + comp_c_frag_m * 64     ][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 16][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 32][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 48][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * BM + comp_c_frag_m * 64     ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 16][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 32][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * BM + comp_c_frag_m * 64 + 48][16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * BN + comp_c_frag_n * 64     ][ 0], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 16][ 0], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 32][ 0], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 48][ 0], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * BN + comp_c_frag_n * 64     ][16], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 16][16], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 32][16], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * BN + comp_c_frag_n * 64 + 48][16], BK + BPAD);

    FLOAT2(s_qa[0]) = FLOAT2(sa[load_sa_gmem_m        ]);
    FLOAT2(s_qb[0]) = FLOAT2(sb[load_sb_gmem_n    ]);

    scale_qa0 = s_qa[2 * comp_c_frag_m    ];
    scale_qa1 = s_qa[2 * comp_c_frag_m + 1];

    scale_qb0 = s_qb[2 * comp_c_frag_n    ];
    scale_qb1 = s_qb[2 * comp_c_frag_n + 1];

    scale_qa0f = __half2float(scale_qa0);
    scale_qa1f = __half2float(scale_qa1); 
    scale_qb0f = __half2float(scale_qb0); 
    scale_qb1f = __half2float(scale_qb1);

    load_sa_gmem_m += BSK * NUMQM;
    load_sb_gmem_n += BSK * NUMQN;

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

            scale_ab = scale_a * scale_b;
            wmma::fill_fragment(frag_intc[i][j], static_cast<int32_t>(0.0));
            wmma::mma_sync(frag_intc[i][j], frag_a[0][i], frag_b[0][j], frag_intc[i][j]);
            wmma::mma_sync(frag_intc[i][j], frag_a[1][i], frag_b[1][j], frag_intc[i][j]);

            #pragma unroll
            for(int k=0; k < frag_intc[i][j].num_elements; k++) {
                frag_fpc[i][j].x[k] += scale_ab * frag_intc[i][j].x[k];
            }
        }
    }
    
    __syncthreads();

    // int32_t* ch = reinterpret_cast<int32_t*>(c);
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

    // int32_t* ch = reinterpret_cast<int32_t*>(c);
    int store_sc_gmem_m = by * BSM + comp_c_frag_m * 2;
    int store_sc_gmem_n = bx * BSN + comp_c_frag_n * 2;
    int store_sc_gmem_addr = OFFSET(store_sc_gmem_m, store_sc_gmem_n, NUMQN);
    // printf("tid: %d | by: %d, bx: %d | store_sc_gmem_addr: %d | store_c_gmem_m: %d | store_c_gmem_n: %d | NUMQN: %d \n", 
    //         tid, by, bx, store_sc_gmem_addr, store_c_gmem_m, store_c_gmem_n, NUMQN);


    // output tensor address
    int store_c_smem_m = (tid_mod >> 2) << 2;
    int store_c_smem_n = (tid_mod &  3) << 3;

    float BiasMax = biasmax[0];
    #pragma unroll
    for (int x = 0; x < 4; x += 2) {
        for (int y = 0; y < 4; y += 2) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    wmma::store_matrix_sync(&acc_float[comp_c_frag_m * 32 + i * 16][comp_c_frag_n * 32 + j * 16], frag_fpc[x + i][y + j], 2 * QN + ACCPAD, wmma::mem_row_major);
                }
            }

            float absMaxVal = -1e10;
            float thd_max = 0;
            for (int k = 0; k < QN; k++) {
                thd_max = fmaxf(thd_max, fabs(acc_float[comp_c_frag_m * 32 + k][comp_c_frag_n * 32 + tid_mod]));
                // if ((tid >= 32) && (tid < 64) && (x == 2) && (y == 0)) printf("tid: %d, k: %d, xx: %d, yy: %d | thd_max = %f \n", tid, k, comp_c_frag_m * 32 + k, comp_c_frag_n * 32 + tid_mod, thd_max);
            }

            // if (tid == 32) printf("tid: %d, xx: %d, yy: %d, thd_max: %f \n", tid, comp_c_frag_m, comp_c_frag_n * 32 + tid_mod, thd_max);
            thread_max[comp_c_frag_m][comp_c_frag_n * 32 + tid_mod] = thd_max;
            __syncwarp();

            for (int active_thread_num = 16; active_thread_num >= 1; active_thread_num /= 2) { 
                if (tid_mod < active_thread_num) { 
                    thread_max[comp_c_frag_m][comp_c_frag_n * 32 + tid_mod] = fmaxf(thread_max[comp_c_frag_m][comp_c_frag_n * 32 + tid_mod], thread_max[comp_c_frag_m][comp_c_frag_n * 32 + tid_mod + active_thread_num]); 
                }
                __syncwarp(); 
            } 

            absMaxVal = thread_max[comp_c_frag_m][comp_c_frag_n * 32] + BiasMax;

            for (int i = 0; i < 4; i++) {
                int8_t ci_val[8];
                float cf_val[8];
                float bias_reg[8];
                
                FLOAT4(cf_val[0]) = FLOAT4(acc_float[comp_c_frag_m * 32 + store_c_smem_m + i][comp_c_frag_n * 32 + store_c_smem_n + 0]);
                FLOAT4(cf_val[4]) = FLOAT4(acc_float[comp_c_frag_m * 32 + store_c_smem_m + i][comp_c_frag_n * 32 + store_c_smem_n + 4]);
                // FLOAT4(cf_val[4]) = FLOAT4(acc_float[comp_c_frag_m * 32 + store_c_smem_m + i][comp_c_frag_n * 32 + store_c_smem_n + 4]);

                FLOAT4(bias_reg[0]) = FLOAT4(bias[store_c_gmem_n + y * 16 + store_c_smem_n + 0]);
                FLOAT4(bias_reg[4]) = FLOAT4(bias[store_c_gmem_n + y * 16 + store_c_smem_n + 4]);

                for (int j = 0; j < 8; j++) {
                    // printf("!!!%d %d %d %f %f\n", __float2int_rn(127 * (cf_val[j] + bias_reg[j]) / absMaxVal), clamp(__float2int_rn(127 * (cf_val[j] + bias_reg[j]) / absMaxVal), -127, 127), __float2int_rn(127 * cf_val[j] / absMaxVal), cf_val[j], bias_reg[j]);
                    ci_val[j] = clamprowcol(__float2int_rn(127 * (cf_val[j] + bias_reg[j]) / absMaxVal), -127, 127);
                    // if (tid == 32) printf("tid: %d | c_val: %d | absMaxVal: %f | acc_float: %f | xx: %d | yy: %d | store_xy: %d \n", 
                    //                        tid, c_val, absMaxVal, acc_float[comp_c_frag_m * 32 + store_c_smem_m + i][comp_c_frag_n * 32 + store_c_smem_n + j],
                    //                        comp_c_frag_m * 32 + store_c_smem_m + i, comp_c_frag_n * 32 + store_c_smem_n + j,
                    //                        store_c_gmem_m + (x * 16 + store_c_smem_m + i) * N + (y * 16 + store_c_smem_n + j));
                }
                INT2(c[store_c_gmem_addr + (x * 16 + store_c_smem_m + i) * N + (y * 16 + store_c_smem_n)]) = INT2(ci_val[0]);
            }

            // printf("line: %d, tid: %d, store_sc_gmem_addr: %d, x: %d, y: %d, absMaxVal: %f, Final Address: %d \n", __LINE__, tid, store_sc_gmem_addr, x, y, absMaxVal, store_sc_gmem_addr + (x / 2) * NUMQN + (y / 2));
            sc[store_sc_gmem_addr + (x / 2) * NUMQN + (y / 2)] = __float2half(absMaxVal / 127);
            // printf("%d \n", __LINE__);
        }
    }

    // // int32_t* ch = reinterpret_cast<int32_t*>(c);
    // int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    // int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    // int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     #pragma unroll
    //     for (int j = 0; j < 4; j++) {
    //         wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_fpc[i][j], N, wmma::mem_row_major);
    //     }
    // }
}

// Quantize + Dequantize Vanilla INT8 GEMM
std::tuple<torch::Tensor, torch::Tensor> igemm_output_int_quantize_bias_rowcol_cuda(
    torch::Tensor X, torch::Tensor W, torch::Tensor bias, torch::Tensor biasmax,
    torch::Tensor SX, torch::Tensor SW,
    const int M, const int N, const int K) {
    // X shape (M, K), W shape (K, N)
    
    const int BM = 128, BN = 128;
    dim3 blockDim(128);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    const int NSPLIT = 4096;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    int QM = SX.size(1);
    int QK = SX.size(0);
    int QN = SW.size(1);

    auto option_output = torch::TensorOptions().dtype(torch::kInt8).device(X.device());
    torch::Tensor O = torch::empty({M, N}, option_output);

    auto option_scale_output = torch::TensorOptions().dtype(torch::kFloat16).device(X.device());
    torch::Tensor SO = torch::empty({QM, QN}, option_scale_output);

    // std::cout << X.scalar_type() << std::endl;
    // std::cout << X.dtype() << std::endl;
    // std::cout << O.scalar_type() << std::endl;

    // std::cout << SX << std::endl;
    // std::cout << SW << std::endl;

    int maxSmem = 30 * 1024;
    cudaFuncSetAttribute(igemm_output_int_quantize_bias_rowcol_cuda_kernel<int8_t, int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmem);

    igemm_output_int_quantize_bias_rowcol_cuda_kernel<int8_t, int8_t><<<gridDim, blockDim, maxSmem>>>(
        X.data_ptr<int8_t>(),
        W.data_ptr<int8_t>(),
        bias.data_ptr<float>(),
        biasmax.data_ptr<float>(),
        (half*)SX.data_ptr<at::Half>(),
        (half*)SW.data_ptr<at::Half>(),
        O.data_ptr<int8_t>(),
        (half*)SO.data_ptr<at::Half>(),
        M,
        N,
        K
    );
    return std::make_tuple(O, SO);
}