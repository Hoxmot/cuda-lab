#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

__global__ void kernel(int* tablica, int liczba_elementow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int step = gridDim.x * blockDim.x;

    for (; i < liczba_elementow; i += step) {
	tablica[i] = 2 * tablica[i];
    }
}

int main() {
    const int liczba_elementow = 4096;
    int tablica_cpu[liczba_elementow];

    int* tablica_gpu;
    cudaError_t status;

    for (int i = 0; i < liczba_elementow; i++) {
	tablica_cpu[i] = i;
    }
    
    status = cudaMalloc((void**)&tablica_gpu, sizeof(int) * liczba_elementow);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    status = cudaMemcpy(tablica_gpu, tablica_cpu, sizeof(int) * liczba_elementow, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    kernel<<<2, 256, 0>>>(tablica_gpu, liczba_elementow);

    status = cudaMemcpy(tablica_cpu, tablica_gpu, sizeof(int) * liczba_elementow, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    status = cudaFree(tablica_gpu);
    if (status != cudaSuccess) {
	cout << cudaGetErrorString(status) << endl;
    }

    for (int i = 0; i < 10; i++) {
	cout << "indeks " << i << " wartosc " << tablica_cpu[i] << endl;
    }
    cout << "indeks 4095 wartosc " << tablica_cpu[4095] << endl;

    return 0;
}
