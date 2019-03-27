#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void kernel() {
    
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x;
    int c = gridDim.x;
    int d = gridDim.x * blockDim.x;

    printf("Hello World, my number: %d, block number: %d, blocks: %d, threads: %d\n", a, b, c, d);

}

int main() {
    kernel<<<2, 256, 0>>>();
    cudaDeviceSynchronize();    
    return 0;
}
