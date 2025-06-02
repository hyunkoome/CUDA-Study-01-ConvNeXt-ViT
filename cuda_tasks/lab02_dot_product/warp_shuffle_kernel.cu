#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

extern "C"
__global__ void dotProductWarp(const float* a, const float* b, float* result, int size) {
    float sum = 0.0f;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        sum += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    sum = warpReduceSum(sum);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(result, sum);
}
