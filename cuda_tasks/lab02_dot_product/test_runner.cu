#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N (1 << 20)
#define THREADS 256

// 커널 함수 선언 (각각 별도 .cu 파일에서 정의)
// __global__ void dotProductShared(const float* a, const float* b, float* result, int size);
// __global__ void dotProductDouble(const double* a, const double* b, double* result, int size);
// __global__ void dotProductTensor(const float* a, const float* b, float* result, int size);
// __global__ void dotProductWarp(const float* a, const float* b, float* result, int size);

extern "C" __global__ void dotProductShared(const float* a, const float* b, float* result, int size);
extern "C" __global__ void dotProductDouble(const double* a, const double* b, double* result, int size);
extern "C" __global__ void dotProductTensor(const float* a, const float* b, float* result, int size);
extern "C" __global__ void dotProductWarp(const float* a, const float* b, float* result, int size);


int main() {
    // 호스트 메모리 할당
    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 디바이스 메모리 할당
    float *d_a, *d_b, *d_result_f32;
    double *d_a64, *d_b64, *d_result_f64;
    cudaMalloc(&d_a, sizeof(float) * N);
    cudaMalloc(&d_b, sizeof(float) * N);
    cudaMalloc(&d_result_f32, sizeof(float));
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Double precision용 입력도 준비
    double* h_a64 = new double[N];
    double* h_b64 = new double[N];
    for (int i = 0; i < N; ++i) {
        h_a64[i] = 1.0;
        h_b64[i] = 2.0;
    }
    cudaMalloc(&d_a64, sizeof(double) * N);
    cudaMalloc(&d_b64, sizeof(double) * N);
    cudaMalloc(&d_result_f64, sizeof(double));
    cudaMemcpy(d_a64, h_a64, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b64, h_b64, sizeof(double) * N, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS - 1) / THREADS;
    float result_f32 = 0;
    double result_f64 = 0;
    std::chrono::high_resolution_clock::time_point t1, t2;

    // ------------------------------------
    // [1] Shared Memory
    // ------------------------------------
    cudaMemset(d_result_f32, 0, sizeof(float));
    t1 = std::chrono::high_resolution_clock::now();
    dotProductShared<<<blocks, THREADS>>>(d_a, d_b, d_result_f32, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&result_f32, d_result_f32, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[1] Shared Memory:\t" << result_f32 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // ------------------------------------
    // [2] Double Precision
    // ------------------------------------
    cudaMemset(d_result_f64, 0, sizeof(double));
    t1 = std::chrono::high_resolution_clock::now();
    dotProductDouble<<<blocks, THREADS>>>(d_a64, d_b64, d_result_f64, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&result_f64, d_result_f64, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "[2] Double Precision:\t" << result_f64 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // ------------------------------------
    // [3] Tensor Inner Product (Same as Shared)
    // ------------------------------------
    cudaMemset(d_result_f32, 0, sizeof(float));
    t1 = std::chrono::high_resolution_clock::now();
    dotProductTensor<<<blocks, THREADS>>>(d_a, d_b, d_result_f32, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&result_f32, d_result_f32, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[3] Tensor Inner:\t" << result_f32 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // ------------------------------------
    // [4] Warp Shuffle
    // ------------------------------------
    cudaMemset(d_result_f32, 0, sizeof(float));
    t1 = std::chrono::high_resolution_clock::now();
    dotProductWarp<<<blocks, THREADS>>>(d_a, d_b, d_result_f32, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&result_f32, d_result_f32, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[4] Warp Shuffle:\t" << result_f32 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // 메모리 해제
    delete[] h_a;
    delete[] h_b;
    delete[] h_a64;
    delete[] h_b64;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result_f32);
    cudaFree(d_a64); cudaFree(d_b64); cudaFree(d_result_f64);

    return 0;
}

