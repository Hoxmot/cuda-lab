#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

__global__ void kernel(int* tab, int elem_number) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int step = gridDim.x * blockDim.x;

    for (; i < elem_number; i += step) {
	tab[i] = 2 * tab[i];
    }
}

int main() {
    const int elem_number = 4096;
    int tab_cpu[elem_number];

    int* tab_gpu;
    cudaError_t status;

    for (int i = 0; i < elem_number; i++) {
	tab_cpu[i] = i;
    }
    
    status = cudaMalloc((void**)&tab_gpu, sizeof(int) * elem_number);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    status = cudaMemcpy(tab_gpu, tab_cpu, sizeof(int) * elem_number, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    kernel<<<2, 256, 0>>>(tab_gpu, elem_number);

    status = cudaMemcpy(tab_cpu, tab_gpu, sizeof(int) * elem_number, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    status = cudaFree(tab_gpu);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    for (int i = 0; i < 10; i++) {
	cout << "Index " << i << " value " << tab_cpu[i] << endl;
    }
    cout << "Index 4095 value " << tab_cpu[4095] << endl;

    return 0;
}
