#include <iostream>
#include <cuda_runtime_api.h>
#include <stdlib.h>

#define LEN 2

using namespace std;

__global__ void matrix_mul(double *m1, double *m2, double *m3, size_t len) {

    //__shared__ double shm[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    for (; i < len; i += step) {
        for (int k = 0; k < len; k++) {
            m3[i * LEN + k] = 0;
            for (int j = 0; j < len; j++) {
                m3[i * LEN + k] += m1[i * LEN + j] * m2[j * LEN + k];
            }
        }
    }
}


int main() {

    double *m1_cpu, *m2_cpu, *res_cpu;

    cudaError_t status;
    double *m1_gpu, *m2_gpu, *res_gpu;

    m1_cpu = (double*)calloc(LEN * LEN, sizeof(double));
    m2_cpu = (double*)calloc(LEN * LEN, sizeof(double));

    for (int i = 0; i < LEN; i++) {
        for (int k = 0; k < LEN; k++) {
            m1_cpu[i * LEN + k] = (i + k) + 7;
            m2_cpu[i * LEN + k] = (i + k) + 3;
        }
    }

    for (int i = 0; i < LEN * LEN; i++) {
        cout << m1_cpu[i] << " ";
        if (i % LEN == LEN - 1)
            cout << endl;
    }
    cout << endl;

    for (int i = 0; i < LEN * LEN; i++) {
        cout << m2_cpu[i] << " ";
        if (i % LEN == LEN - 1)
            cout << endl;
    }
    cout << endl;

    status = cudaMalloc((void**)&m1_gpu, sizeof(int) * LEN * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMalloc((void**)&m2_gpu, sizeof(int) * LEN * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMalloc((void**)&res_gpu, sizeof(int) * LEN * LEN);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    status = cudaMemcpy(m1_gpu, m1_cpu, sizeof(int) * LEN * LEN, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaMemcpy(m2_gpu, m2_cpu, sizeof(int) * LEN * LEN, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    free(m1_cpu);
    free(m2_cpu);

    matrix_mul<<<2,2>>>(m1_gpu, m2_gpu, res_gpu, LEN);

    res_cpu = (double*)calloc(LEN * LEN, sizeof(double));

    status = cudaMemcpy(res_cpu, res_gpu, sizeof(int) * LEN * LEN, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    status = cudaFree(m1_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaFree(m2_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }
    status = cudaFree(res_gpu);
    if (status != cudaSuccess) {
	    cout << cudaGetErrorString(status) << endl;
    }

    for (int i = 0; i < LEN * LEN; i++) {
        cout << res_cpu[i] << " ";
        if (i % LEN == LEN - 1)
            cout << endl;
    }

    free(res_cpu);

    return 0;

}