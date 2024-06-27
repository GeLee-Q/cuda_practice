#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// max(0, val)
__global__ void relu(float* x, float*y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) {
        y[idx] = fmaxf(x[idx], 0);
    }
}


__global__ void relu_v2(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx < N){
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.w = reg_x.w;
        reg_y.x = reg_x.x;
        reg_y.y = reg_x.y;
        reg_y.z = reg_x.z;
        FLOAT4(y[idx]) = reg_y;
    }
}