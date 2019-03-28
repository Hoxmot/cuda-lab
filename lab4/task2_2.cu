#include <iostream>
#include <cuda_runtime_api.h>

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

__global__ void mul(int *m, int *v, int *res, size_t l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int k;
    for (; i < l * l; i += step) {
	    k = i / l;
	    res[i] = m[i] * v[k];
    }
}

__global__ void gen_numbers(int *m, int *v, size_t l) {
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

__global__ void print_numbers(int *res, size_t l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (; i < l * l; i += step) {
        printf("%d ", res[i]);
        if (i % l == l - 1) {
            printf("\n");
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

    gen_input<<<2, 10, 0>>>(m_gpu, v_gpu, LEN);

    add_vec<<<2, 10, 0>>>(m_gpu, v_gpu, res_gpu, LEN);
 
    print_numbers<<<2, 10, 0>>>(res_gpu, l);

    status = cudaMemcpy(res_cpu, res_gpu, sizeof(int) * LEN * LEN, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

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
