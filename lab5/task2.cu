/*
  modified from source:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
*/

#include <iostream>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 32

#ifndef LEN
#error LEN is not defined
#endif

using namespace std;

typedef struct {
    int width;
    int height;
    int stride;
    double *elements;
} Matrix;

__device__ double get_element(const Matrix M, int row, int col) {
    return M.elements[row * M.stride + col];
}

__device__ void set_element(Matrix M, int row, int col,
                            double value) {
    M.elements[row * M.stride + col] = value;
}

__device__ Matrix get_sub_matrix(Matrix M, int row, int col) {
    Matrix Msub;
    Msub.width = BLOCK_SIZE;
    Msub.height = BLOCK_SIZE;
    Msub.stride = M.stride;
    Msub.elements = &M.elements[M.stride * BLOCK_SIZE * row
                                + BLOCK_SIZE * col];
    return Msub;
}

__global__ void matrix_mul(const Matrix A, const Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = get_sub_matrix(C, blockRow, blockCol);

    double c_value = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        Matrix Asub = get_sub_matrix(A, blockRow, m);
        Matrix Bsub = get_sub_matrix(B, m, blockCol);

        for (int e = 0; e < BLOCK_SIZE; ++e)
            c_value += Asub[row * Asub.stride + e] * Bsub[e * Bsub.stride + col];

    }

    set_element(Csub, row, col, c_value);

}

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

void matrix_mul_cpu(const Matrix M1, const Matrix M2, Matrix M3) {
    Matrix M1_gpu, M2_gpu, M3_gpu;

    M1_gpu.width = M1_gpu.stride = M1.width;
    M1_gpu.height = M1.height;
    ssize_t size = M1.width * M1.height * sizeof(double);
    handleCudaMalloc((void**)&M1_gpu.elements, size);
    handleCudaMemcpy(M1_gpu.elements, M1.elements, size, cudaMemcpyHostToDevice);

    M2_gpu.width = M2_gpu.stride = M2.width;
    M2_gpu.height = M2.height;
    size = M2.width * M2.height * sizeof(double);
    handleCudaMalloc((void**)&M2_gpu.elements, size);
    handleCudaMemcpy(M2_gpu.elements, M2.elements, size, cudaMemcpyHostToDevice);

    M3_gpu.width = M3_gpu.stride = M3.width;
    M3_gpu.height = M3.height;
    size = M3.height * M3.width * sizeof(double);
    handleCudaMalloc((void**)&M3_gpu.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(M2.width / dimBlock.x, M1.height / dimBlock.y);
    matrix_mul<<<dimGrid, dimBlock>>>(M1_gpu, M2_gpu, M3_gpu);

    handleCudaMemcpy(M3.elements, M3_gpu.elements, size, cudaMemcpyDeviceToHost);

    handleCudaFree(M1_gpu.elements);
    handleCudaFree(M2_gpu.elements);
    handleCudaFree(M3_gpu.elements);
}

int main() {

    Matrix M1_cpu, M2_cpu, ResCpu;
    
    M1_cpu.stride = M1_cpu.height = M1_cpu.width = LEN;
    M1_cpu.elements = (double*)calloc(LEN * LEN, sizeof(double));

    M2_cpu.stride = M2_cpu.width = M2_cpu.height = LEN; 
    M2_cpu.elements = (double*)calloc(LEN * LEN, sizeof(double));

    for (int i = 0; i < LEN; i++) {
        for (int k = 0; k < LEN; k++) {
            M1_cpu.elements[i * LEN + k] = (i + k) + 7;
            M2_cpu.elements[i * LEN + k] = (i + k) + 3;
        }
    }

    ResCpu.height = ResCpu.stride = ResCpu.width = LEN;
    ResCpu.elements = (double*)calloc(LEN * LEN, sizeof(double));

    matrix_mul_cpu(M1_cpu, M2_cpu, ResCpu);

    free(M1_cpu.elements);
    free(M2_cpu.elements);
    free(ResCpu.elements);

    return 0;

}