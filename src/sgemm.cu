#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

//SGEMM:
// block tile (BM, BK) + K Tile
__global__ void sgemm(float* a, float* b, float *c, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    // 计算单次 thread 搬运到smem的 x, y -> load_smem_a_m, load_smem_b_k
    int load_smem_a_m = tid / 32; // tid / BM, 
    int load_smem_a_k = tid % 32; // tid % BK
    int load_smem_b_k = tid / 32; // tid / BK
    int load_smem_b_n = tid % 32; // tid % BN

    // 计算单次 thread 处理的 global mem 的 数据
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    float sum = 0.f;
    for(int bk = 0; bk < (K + BK - 1)/BK; ++bk ) {
        // 计算 global mem 的取数据的 addr
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];

        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gemm_b_addr = load_gmem_b_k * N +  load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gemm_b_addr];
        __syncthreads();
        #pragma unroll
        for(int k = 0; k < BK; ++k) {
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }

    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;
}





__global__ void nativeSgemm(float* a, float* b, float *c, int M, int N, int K){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if(m < M && n < N) {
        float sum = 0.0;
        for(int k = 0; k < K; k++){
            sum += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = sum;
    }
}

__global__ void sgemm_v2(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;

    int load_smem_a_m = tid / BM;
    int load_smem_a_k = tid % BK;
    int load_smem_b_k = tid / BK;
    int load_smem_b_n = tid % BN;

    int load_gmem_a_m = by * BM + load_smem_a_m; // row idx
    int load_gmem_b_n = bx * BN + load_smem_b_n; // col idx

    float sum = 0.f;
    for (int bk = 0; bk < ( K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k ;
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
        
        int load_gmem_b_k = bk * BK + load_smem_b_k ;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();

        for(int k = 0; k < bk; k++) {
            int com_smem_m = load_smem_a_m;
            int com_smem_n = load_smem_b_n;
            sum += s_a[com_smem_m][k] * s_b[k][com_smem_n];
        }
        __syncthreads();
    }

    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_c_gemm_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_c_gemm_addr] = sum;

}