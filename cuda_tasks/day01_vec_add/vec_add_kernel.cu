// vec_add_kernel.cu

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vec_add(float *a, float *b, float *c, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

// ✅ extern "C" 추가 필요
extern "C" void vec_add_launcher(float* a, float* b, float* c, int N) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    vec_add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
