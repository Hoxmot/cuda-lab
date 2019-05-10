#include <iostream>
#include <cuda_runtime_api.h>
#include <stdio.h>

using namespace std;

#ifndef LEN
#error LEN is not defined
#endif

#ifndef THREADS
#error THREADS is not defined
#endif

#ifndef BLOCKS
#error BLOCKS is not defined
#endif

__global__ void mul(int *m, int *v, int *res, size_t l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int k;
    for (; i < l * l; i += step) {
	    k = i / l;
	    res[i] = m[i] * v[k];
    }
}

__global__ void gen_input(int *m, int *v, size_t l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int k;
    for (; i < l * l; i += step) {
        m[i] = i + 42;
        if (i % l == 0) {
            k = i / l;
            v[k] = k * 40 + 2;
        }
    }
}

__global__ void print_numbers(int *res, size_t l, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    if (dim == 2) {
        for (; i < l * l; i += step) {
            printf("%d ", res[i]);
            if (i % l == l - 1) {
                printf("\n");
            }
        }
    }
    if (dim == 1) {
        for (; i < l; i += step) {
            printf("%d ", res[i]);
        }
    }
}

int main() {
    cudaError_t status;
    int *m_gpu, *v_gpu, *res_gpu;

    status = cudaMalloc((void**)&m_gpu, sizeof(int) * LEN * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMalloc((void**)&v_gpu, sizeof(int) * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMalloc((void**)&res_gpu, sizeof(int) * LEN * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    gen_input<<<2, 32, 0>>>(m_gpu, v_gpu, LEN);

    mul<<<BLOCKS, THREADS, 0>>>(m_gpu, v_gpu, res_gpu, LEN);
 
    status = cudaFree(m_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaFree(v_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaFree(res_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    return 0;

}
