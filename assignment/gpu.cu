#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N   100
#define N2  128
#define T_NUMBER  9    // (360 - 270) / 10 -> number of different T values
#define TN  900  // N * T_NUMBER -> number of different T values times N
#define STEPS 1000

/* macro from: https://gist.github.com/NicholasShatokhin/3769635 */
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
    return EXIT_FAILURE;}} while(0)

__device__ float Energy(float* positionX, float* positionY, float* positionZ, int i) {
    
    int idx = threadIdx.x;

    // I'm using vector of size being power of 2
    // This way reduction is easier
    __shared__ float vec_E[N2];
    
    // Initializing unused values to 0
    if (idx < N2 - N)
        vec_E[N + idx] = 0;

	float E = 0;

    float X = positionX[i] - positionX[idx];
    float Y = positionY[i] - positionY[idx];
    float Z = positionZ[i] - positionZ[idx];

    vec_E[idx] = (i != idx ? (1. / (X*X + Y*Y + Z*Z)) : 0);
    
    __syncthreads();

    for (unsigned int s = N2 / 2; s > 0; s >>= 1) {
        if (idx < s && idx + s < N) {
            vec_E[idx] += vec_E[idx + s];
        }
        __syncthreads();
    }

    E = vec_E[0];

    return E;
}

__device__ float RAND1(curandState* state) {
	return (curand_uniform(state) - 0.5); 
}
__device__ float RAND0(curandState* state) {
	return curand_uniform(state);
}

__device__ void makemove(float* positionX, float* positionY, float* positionZ, float* positionNEWX, float* positionNEWY, float* positionNEWZ) {
    int idx = threadIdx.x;
    positionX[idx] = positionNEWX[idx];
    positionY[idx] = positionNEWY[idx];
    positionZ[idx] = positionNEWZ[idx];
}

__device__ void newpos(float* positionNEWX, float* positionNEWY, float* positionNEWZ, int i, float size, curandState *state) {
	positionNEWX[i] += RAND1(state);
    
    if (positionNEWX[i] < 0)
        positionNEWX[i] = fabsf(positionNEWX[i]);
    else if (positionNEWX[i] > size)
        positionNEWX[i] -= size;
    
    positionNEWY[i] += RAND1(state);
    
    if (positionNEWY[i] < 0)
        positionNEWY[i] = fabsf(positionNEWY[i]);
    else if (positionNEWY[i] > size)
        positionNEWY[i] -= size;
    
    positionNEWZ[i] += RAND1(state);
    
    if (positionNEWZ[i] < 0)
        positionNEWZ[i] = fabsf(positionNEWZ[i]);
    else if (positionNEWZ[i] > size)
        positionNEWZ[i] -= size;
}

void pr(float* positionX, float* positionY, float* positionZ) {
	int i;
	for (i = 0; i < N; i++)
		printf("%f %f %f\n", positionX[i], positionY[i], positionZ[i]);
	printf("\n\n");
}

// TODO
__global__ void simulate(float* positionX, float* positionY, float* positionZ, float* stepY, unsigned long seed) {
    int idx = threadIdx.x;
    int offset = blockIdx.x * N;

    // TODO: check CUDA cbrt
    float size = cbrtf(N);
 
    __shared__ float shrX[N], shrY[N], shrZ[N];
    __shared__ float newX[N], newY[N], newZ[N];
    __shared__ float rnd;

    int sY = 0;
    int T = 270 + offset * 10;
    float kT = .01/T;
    float E;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // start
    // TODO: state
    shrX[idx] = RAND0(&state) * size;
    shrY[idx] = RAND0(&state) * size;
    shrZ[idx] = RAND0(&state) * size;
    newX[idx] = shrX[idx];
    newY[idx] = shrY[idx];
    newZ[idx] = shrZ[idx];
    __syncthreads();

    for (int k = 0; k < STEPS; k++) {
        for (int i = 0; i < N; i++) {
            // HERE
            if (idx == i)
                newpos(newX, newY, newZ, i, size, &state);
            __syncthreads();

            E = Energy(newX, newY, newZ, i) - Energy(shrX, shrY, shrZ, i);
            if (idx == i)
                rnd = RAND0(&state);
            __syncthreads();

            if (E < 0) {
                makemove(shrX, shrY, shrZ, newX, newY, newZ);
                if (idx == 0)
                    sY++;
            }
            else if(rnd < expf(-E/kT)){
                makemove(shrX, shrY, shrZ, newX, newY, newZ);
                if (idx == 0)
                    sY++;
            }
            __syncthreads();
        }
    }

    positionX[offset + idx] = shrX[idx];
    positionY[offset + idx] = shrY[idx];
    positionZ[offset + idx] = shrZ[idx];
    if (idx == 0)
        stepY[blockIdx.x] = sY;
}

int main() {
     
    float *positionX_gpu, *positionY_gpu, *positionZ_gpu, *stepY_gpu;

    CUDA_CALL(cudaMalloc((void**)&positionX_gpu, sizeof(float) * TN));
    CUDA_CALL(cudaMalloc((void**)&positionY_gpu, sizeof(float) * TN));
    CUDA_CALL(cudaMalloc((void**)&positionZ_gpu, sizeof(float) * TN));
    CUDA_CALL(cudaMalloc((void**)&stepY_gpu, sizeof(float) * T_NUMBER));

    // TODO: setpu kernel size and call it

    dim3 dimBlock(N);
    dim3 dimGrid(T_NUMBER);
    simulate<<<dimGrid, dimBlock>>>(positionX_gpu, positionY_gpu, positionZ_gpu, stepY_gpu, time(NULL));
    
    float* positionX = (float*) malloc(TN * sizeof(float));
    float* positionY = (float*) malloc(TN * sizeof(float));
    float* positionZ = (float*) malloc(TN * sizeof(float));
    float* stepY = (float*) malloc(T_NUMBER * sizeof(float));

    CUDA_CALL(cudaMemcpy(positionX, positionX_gpu, sizeof(float) * TN, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(positionY, positionY_gpu, sizeof(float) * TN, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(positionZ, positionZ_gpu, sizeof(float) * TN, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(stepY, stepY_gpu, sizeof(float) * T_NUMBER, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(positionX_gpu));
    CUDA_CALL(cudaFree(positionY_gpu));
    CUDA_CALL(cudaFree(positionZ_gpu));
    CUDA_CALL(cudaFree(stepY_gpu));

    int T;
    for (int i = 0; i < T_NUMBER; ++i) {
        T = 270 + i * 10;
        printf("Stepe ACC %d  %f\n", T, stepY[i] * 1./STEPS);
    }

    free(positionX);
    free(positionY);
    free(positionZ);
    free(stepY);

    return 0;
}

