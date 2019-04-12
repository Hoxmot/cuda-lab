#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

#define VEC_NUM 100
#define VEC_SIZE 8192
#define BLOCK_SIZE 32
#define THREADS 256  // = 8192 / BLOCK_SIZE

#ifdef DEBUG
static const int debug = 1;
#else
static const int debug = 0;
#endif

__global__ void mul(double *matrix, double *vec) {
    ssize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t row_beg = row * VEC_SIZE;
    for (int i = 0; i < VEC_SIZE; ++i) {
	    matrix[row_beg + i] = matrix[row_beg + i] * vec[row];
    }
}

__global__ void gen_matrix(double *matrix) {

    ssize_t i = blockIdx.x * blockDim.x + threadIdx.x;
    i *= VEC_SIZE;
    for (; i < VEC_SIZE; ++i) {
        // TODO: generation of easier numbers
	    matrix[i] = i * 42 + 2137;
    }

}

int main() {
    double *vec_cpu, *matrix_cpu;

    cudaError_t status;
    double *vec_gpu, *matrix_gpu;

    status = cudaMalloc((void**)&matrix_gpu, VEC_SIZE * VEC_SIZE * sizeof(double));
    if (status != cudaSuccess) {
	    printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaMalloc((void**)&vec_gpu, VEC_SIZE * sizeof(double));
    if (status != cudaSuccess) {
	    printf("%s\n", cudaGetErrorString(status));
    }

    gen_matrix<<<BLOCK_SIZE, THREADS>>>(matrix_gpu);

    for (double i = 0 ; i < VEC_NUM; ++i) {
        vec_cpu = (double*)calloc(VEC_SIZE, sizeof(double));

        for (int j = 0; j < VEC_SIZE; ++j) {
            // TODO: generation of easier numbers
            vec_cpu[j] = (j + i + 42) / 2137;
        }

        status = cudaMemcpy(vec_gpu, vec_cpu, VEC_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(status));
        }

        mul<<<BLOCK_SIZE, THREADS>>>(matrix_gpu, vec_gpu);

        free(vec_cpu);
    }

    matrix_cpu = (double*)calloc(VEC_SIZE * VEC_SIZE, sizeof(double));

    status = cudaMemcpy(matrix_cpu, matrix_gpu, VEC_SIZE * VEC_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaFree(vec_gpu);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaFree(matrix_gpu);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    if (debug) {
        // TODO: Comparison of results
        // 1. generate result on cpu
        // 2. compare all the elements
        // maybe some formula for each element?
    }

    free(matrix_cpu);

    return 0;

}