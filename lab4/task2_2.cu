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
	k = i / l
	res[i] = m[i] * v[k];
    }
}

int main() {
    int m_cpu[LEN * LEN], v_cpu[LEN], res_cpu[LEN * LEN];

    for (int i = 0; i < LEN; i++) {
	v1_cpu[i] = i;
	v2_cpu[i] = i * 40 + 2;
    }

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

    status = cudaMemcpy(m_gpu, m_cpu, sizeof(int) * LEN * LEN, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMemcpy(v_gpu, v_cpu, sizeof(int) * LEN, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    add_vec<<<2, 10, 0>>>(m_gpu, v_gpu, res_gpu, LEN);
 
    status = cudaMemcpy(res_cpu, res_gpu, sizeof(int) * LEN * LEN, cudaMemcpyDeviceToHost);
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

    for (int i = 0; i < LEN; i++) {
	for (int k = 0; k < LEN; k++) {
	    cout << res_cpu[i * LEN + k] << " ";
	}
	cout << endl;
    }

    return 0;

}
