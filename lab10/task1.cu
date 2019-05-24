#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define FULL_MASK   0xffffffff
#define LEN         32

__global__ void reduce32(double* vec, double* res) {
    double val = vec[threadIdx.x];

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    if (threadIdx.x == 0) {
        *res = val;
    }
}

int main() {
    double *vec, *res;
    
    vec = (double*)calloc(LEN, sizeof(double));
    res = (double*)malloc(sizeof(double));

    for (int i = 0; i < LEN; i++) {
        vec[i] = (i + 1) / 42.;
    }

    cudaError_t status;
    double *vec_gpu, *res_gpu;
    
    status = cudaMalloc((void**)&vec_gpu, LEN * sizeof(double));
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }
    status = cudaMalloc((void**)&res_gpu, sizeof(double));
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaMemcpy(vec_gpu, vec, LEN * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    reduce32<<<1, 32>>>(vec_gpu, res_gpu);

    status = cudaMemcpy(res, res_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    printf("%f\n", *res);

    free(vec);
    free(res);
    return 0;
}