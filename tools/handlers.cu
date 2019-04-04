#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

void handleCudaMalloc(void **var, ssize_t size) {
    cudaError_t status;
    status = cudaMalloc(var, size);
    if (status != cudaSuccess) {
	    printf("%s\n", cudaGetErrorString(status));
    }
}

void handleCudaMemcpy(void* dst, const void* src, ssize_t size, cudaMemcpyKind kind) {
    cudaError_t status;
    status = cudaMemcpy(dst, src, size, kind);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }
}

void handleCudaFree(void* pointer) {
    cudaError_t status;
    status = cudaFree(pointer);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }
}