#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

//SGEMM:
__global__ void sgemm(float* a, float* b, float *c, int M, int N, int K) {
    
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