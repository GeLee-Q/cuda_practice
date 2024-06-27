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
        reg_y.x = fmaxf(reg_x.x,0);
        reg_y.y = fmaxf(reg_x.y,0);
        reg_y.z = fmaxf(reg_x.z,0);
        reg_y.w = fmaxf(reg_x.w,0);
        FLOAT4(y[idx]) = reg_y;
    }
}