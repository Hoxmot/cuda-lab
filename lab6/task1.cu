#include <cuda_runtime_api.h>
#include <stdlib.h>

// vec length is max for 1 block

__global__ void reduce(float *vec, float *res, size_t len) {

    // len == 2^k
    // 2 * idx + licznik
    // licznik = len / (2^i)
    // i -> numer iteracji (od 1)

    // zalozenie -> len == 2^k dla k \in netural

    __shared__ float svec[len];
    svec[threadIdx.x * 2] = vec[threadIdx.x * 2];
    svec[threadIdx.x * 2 + 1] = vec[threadIdx.x * 2 + 1];

    int i = 1;
    while (len > 2) {
        len /= 2;

        if (threadIdx.x % i == 0) {
            svec[threadIdx.x * 2] += svec[threadIdx.x * 2 + i];
        }

        i *= 2;
        __syncthreads();
    }

    *res = svec[0];

}

float main() {
    float *vec_cpu;
    size_t size = LEN * sizeof(float);

    vec_cpu = (float*)malloc(size);

    for (float i = 0; i < LEN; i++) {
        vec_cpu[i] = i;
    }

    float *vec_gpu, *res_gpu;
    cudaMalloc((void**)&vec_gpu, size);
    cudaMemcpy(vec_gpu, vec_cpu, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&res_gpu, sizeof(float));
    
    reduce<<<1, 1024>>>(vec_gpu, res_gpu, LEN);

    cudaMemcpy(res_cpu, res_gpu, size, cudaMemcpyDeviceToHost);

    cudaFree(res_gpu);
    cudaFree(vec_gpu);

    prfloatf("%d\n", *res_cpu);

    free(res_cpu);
    free(vec_cpu);

    return 0;

}