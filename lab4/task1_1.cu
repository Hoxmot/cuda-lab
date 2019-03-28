#include <iostream>
#include <cuda_runtime_api.h>
#include <stdlib.h>

using namespace std;

#define LEN 10

__global__ void add_vec(int *v1, int *v2, int *res, size_t l) {
    // cudaError_t status;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    for (; i < l; i+= step) {
	    res[i] = v1[i] + v2[i];
    }

}

int main() {
    int *v1_cpu, *v2_cpu, *res_cpu;

    v1_cpu = calloc(sizeof(int), LEN);
    v2_cpu = calloc(sizeof(int), LEN);

    for (int i = 0; i < LEN; i++) {
	    v1_cpu[i] = i;
	    v2_cpu[i] = i * 40 + 2;
    }

    cudaError_t status;
    int *v1_gpu, *v2_gpu, *res_gpu;

    status = cudaMalloc((void**)&v1_gpu, sizeof(int) * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMalloc((void**)&v2_gpu, sizeof(int) * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMalloc((void**)&res_gpu, sizeof(int) * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    status = cudaMemcpy(v1_gpu, v1_cpu, sizeof(int) * LEN, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMemcpy(v2_gpu, v2_cpu, sizeof(int) * LEN, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    free(v1_cpu);
    free(v2_cpu);

    add_vec<<<2, 10, 0>>>(v1_gpu, v2_gpu, res_gpu, LEN);

    res_cpu = calloc(sizeof(int), LEN);
 
    status = cudaMemcpy(res_cpu, res_gpu, sizeof(int) * LEN, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    status = cudaFree(v1_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaFree(v2_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaFree(res_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    free(res_cpu);

    return 0;

}
