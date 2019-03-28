#include <iostream>
#include <cuda_runtime_api.h>
#include <stdlib.h>

using namespace std;

#define LEN 10

__global__ void mul(int *m, int *v, int *res, size_t l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int k;
    for (; i < l * l; i += step) {
	    k = i / l;
	    res[i] = m[i] * v[k];
    }
}

int main() {
    int *m_cpu, *v_cpu, *res_cpu;

    m_cpu = (int*) calloc(LEN * LEN, sizeof(int));
    v_cpu = (int*) calloc(LEN, sizeof(int));

    for (int i = 0; i < LEN; i++) {
	    for (int k = 0; k < LEN; k++) {
	        m_cpu[i * LEN + k] = (i * LEN + k) + 42;
	    }
	    v_cpu[i] = i * 40 + 2;
    }

    for (int i = 0; i < LEN; i++) {
	    for (int k = 0; k < LEN; k++) {
	        cout << m_cpu[i * LEN + k] << " ";
	    }
	    cout << endl;
    }

    cout << endl;

    for (int i = 0; i < LEN; i++) {
	    cout << v_cpu[i] << " ";
    }

    cout << endl;

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

    free(m_cpu);
    free(v_cpu);

    mul<<<2, 10, 0>>>(m_gpu, v_gpu, res_gpu, LEN);
 
    res_cpu = (int*) calloc(LEN * LEN, sizeof(int));

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

    for (int i = 0; i < LEN; i++) {
	    for (int k = 0; k < LEN; k++) {
	        cout << res_cpu[i * LEN + k] << " ";
	    }
	    cout << endl;
    }

    free(res_cpu);

    return 0;

}
