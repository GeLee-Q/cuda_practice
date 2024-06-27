#include <cstdio>
#include <cmath>
#include <random>

#include <relu.cu>


bool arrays_equal(const float* a, const float* b, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void relu_launcher(float* x, float* y, int N) {
    int blockSize = 256; // 每个线程块中的线程数
    int numBlocks = (N + blockSize - 1) / blockSize; // 计算需要的线程块数量

    float* d_x;
    float* d_y;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // relu<<<numBlocks, blockSize>>>(d_x, d_y, N);
    relu_v2<<<numBlocks, blockSize>>>(d_x, d_y, N);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {

    const int N = 10;
    float h_x[N] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -1.5f, 1.5f, -2.0f, 2.0f, 0.0f};
    float h_y[N] = {0.0f};

    // 期望输出结果
    float expected_y[N] = {0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 0.0f, 1.5f, 0.0f, 2.0f, 0.0f};

    // 调用 relu_launcher 函数
    relu_launcher(h_x, h_y, N);

    if (arrays_equal(h_y, expected_y, N)) {
        printf("Test passed!\n");
    } else {
        printf("Test failed!\n");
        for (int i = 0; i < N; ++i) {
            printf("h_y[%d] = %f, expected_y[%d] = %f\n", i, h_y[i], i, expected_y[i]);
        }
    }

    return 0;
}