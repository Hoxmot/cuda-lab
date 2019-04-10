
#include "limit.h"

__global__ void reduce(int *g_idata, int *g_odata) {
    __shared__ int32_t sdata[SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[i] = g_idata[i];
    __syncthreads();

    for (uint32_t s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}