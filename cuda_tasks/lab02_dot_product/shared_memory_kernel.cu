#include <cuda_runtime.h>

#define THREADS 256

extern "C"
__global__ void dotProductShared(const float* a, const float* b, float* result, int size) {
    __shared__ float cache[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0;
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIdx] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
    }

    if (cacheIdx == 0)
        atomicAdd(result, cache[0]);
}
