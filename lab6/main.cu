#include <cuda_runtime_api.h>
#include <stdlib.h>

#include "reduce.h"
#include "limit.h"

#define BLOCK_SIZE 512

int main () {

    int *data_cpu, *ret_cpu;

    data_cpu = (int*)calloc(SIZE, sizeof(int));

    int *data_gpu, *ret_gpu;
    cudaError_t status;

    status = cudaMalloc((void**)&data_gpu, SIZE * sizeof(int));
    if (status != cudaSuccess) {
	    printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaMalloc((void**)&ret_gpu, SIZE * sizeof(int));
    if (status != cudaSuccess) {
	    printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaMemcpy(data_gpu, data_cpu, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    // call reduce
    reduce<<<>>>(data_cpu, ret_cpu);

    status = cudaFree(data_gpu);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaMemcpy(ret_cpu, ret_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }

    status = cudaFree(ret_gpu);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }



    return 0;
}