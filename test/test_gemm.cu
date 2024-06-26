#include <iostream>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <ticktock.h>
#include <helper_cuda.h>

#include <sgemm.cu>


// compare res 
bool compareMatrices(const float *A, const float *B, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > 1e-3) {
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // allocate host mem
    float *h_A, *h_B, *h_C_myGemm, *h_C_cublas;
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C_myGemm = new float[M * N];
    h_C_cublas = new float[M * N];

    // init 
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = distribution(generator);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = distribution(generator);
    }


    // allocate device mem
    float *d_A, *d_B, *d_C_myGemm, *d_C_cublas;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C_myGemm, M * N * sizeof(float));
    cudaMalloc(&d_C_cublas, M * N * sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // use sgemm
    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    TICK(my_mul);
    // nativeSgemm<<<grid, block>>>(d_A, d_B, d_C_myGemm, M, N, K);
    // sgemm<<<grid, block>>>(d_A, d_B, d_C_myGemm, M, N, K);
    sgemm_v2<<<grid, block>>>(d_A, d_B, d_C_myGemm, M, N, K);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(my_mul);

    // use cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    TICK(cublas);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    checkCudaErrors(cudaDeviceSynchronize());            
    TOCK(cublas);            

    // copy res from device to host
    cudaMemcpy(h_C_myGemm, d_C_myGemm, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // compare results
    if (compareMatrices(h_C_myGemm, h_C_cublas, M * N)) {
        std::cout << "Accuracy test passed!" << std::endl;
    } else {
        std::cout << "Accuracy test failed!" << std::endl;
    }

    // clear mem 
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_myGemm;
    delete[] h_C_cublas;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_myGemm);
    cudaFree(d_C_cublas);
    cublasDestroy(handle);

    return 0;
}