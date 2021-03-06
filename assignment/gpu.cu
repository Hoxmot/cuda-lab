#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// I'm hardcoding all the constants
// It'll be easier to use them instead of sending them as arguments
// over and over again...
#define N         100   // Number of atoms
#define N2        128   // Nearest power of 2 >=N
#define T_NUMBER  9     // (360 - 270) / 10 -> number of different T values
#define TN        900   // N * T_NUMBER -> number of different T values times N
                        // It's the size of arrays we have to use
#define STEPS     1000  // Number of steps

/* macro from: https://gist.github.com/NicholasShatokhin/3769635 */
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
    return EXIT_FAILURE;}} while(0)

__device__ float Energy(float* positionX, float* positionY, float* positionZ, int i) {
    
    int idx = threadIdx.x;

    // I'm using vector of size being power of 2
    // This way reduction is easier to write
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
        if (idx < s) {
            vec_E[idx] += vec_E[idx + s];
        }
        __syncthreads();
    }

    E = vec_E[0];

    return E;
}

// I hope I've used correct functions for random values
__device__ float RAND1(curandState* state) {
	return (curand_uniform(state) - 0.5); 
}
__device__ float RAND0(curandState* state) {
	return curand_uniform(state);
}

__device__ void sync_pos(float* positionX, float* positionY, float* positionZ, float* positionNEWX, float* positionNEWY, float* positionNEWZ) {
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

__global__ void simulate(float* positionX, float* positionY, float* positionZ, float* stepY, unsigned long seed) {
    int idx = threadIdx.x;
    int offset = blockIdx.x * N;

    // I don't have to compute the function multiple times
    __shared__ float size;
    if (idx == 0)
        size = cbrtf(N);
 
    __shared__ float shrX[N], shrY[N], shrZ[N];
    __shared__ float newX[N], newY[N], newZ[N];
    __shared__ float rnd;

    __shared__ float tmpX[N], tmpY[N], tmpZ[N];

    int sY = 0;
    float acc_sY = 0;
    int T = 270 + offset * 10;
    float kT = .01/T;
    float E;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // start
    shrX[idx] = RAND0(&state) * size;
    shrY[idx] = RAND0(&state) * size;
    shrZ[idx] = RAND0(&state) * size;
    sync_pos(newX, newY, newZ, shrX, shrY, shrZ);
    sync_pos(tmpX, tmpY, tmpZ, shrX, shrY, shrZ);
    __syncthreads();

    // These 2 for loops are a bottleneck
    // I can't execute them in parallel
    // The simulation is executed step by step
    // Atom by atom
    for (int k = 0; k < STEPS; k++) {
        // step
        // I'm calculating new position into temporary variables for all the threads
        // 0.01s of improvement, much wow
        newpos(tmpX, tmpY, tmpZ, idx, size, &state);
        for (int i = 0; i < N; i++) {
            if (idx == i) {
                // When i is correct, I move values from tmp to new
                newX[i] = tmpX[i];
                newY[i] = tmpY[i];
                newZ[i] = tmpZ[i];
                // before:
                // newpos(newX, newY, newZ, i, size, &state);
            }
            __syncthreads();

            E = Energy(newX, newY, newZ, i) - Energy(shrX, shrY, shrZ, i);

            // I'll have to use random value in order to check whether I should make a move 
            // I can't comput RAND0() for each thread, because then the if statement may or may not fail
            // This approach allows me to make sure the outcome is the same for all the threads
            if (idx == i)
                rnd = RAND0(&state);  // rnd is shared
            __syncthreads();

            if (E < 0) {
                sync_pos(shrX, shrY, shrZ, newX, newY, newZ);
                if (idx == 0)
                    sY++;
            }
            else if(rnd < expf(-E/kT)){
                sync_pos(shrX, shrY, shrZ, newX, newY, newZ);
                if (idx == 0)
                    sY++;
            }
            __syncthreads();
        }
        acc_sY += sY * 1. / N;
        sY = 0;
    }

    positionX[offset + idx] = shrX[idx];
    positionY[offset + idx] = shrY[idx];
    positionZ[offset + idx] = shrZ[idx];
    if (idx == 0)
        stepY[blockIdx.x] = acc_sY;
}

int main() {
     
    float *positionX_gpu, *positionY_gpu, *positionZ_gpu, *stepY_gpu;

    CUDA_CALL(cudaMalloc((void**)&positionX_gpu, sizeof(float) * TN));
    CUDA_CALL(cudaMalloc((void**)&positionY_gpu, sizeof(float) * TN));
    CUDA_CALL(cudaMalloc((void**)&positionZ_gpu, sizeof(float) * TN));
    CUDA_CALL(cudaMalloc((void**)&stepY_gpu, sizeof(float) * T_NUMBER));

    // block for every T
    // thread for every atom
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

    // I'm not using positionX, positionY, and positionZ
    // But it can be used in some next simulations/operations

    free(positionX);
    free(positionY);
    free(positionZ);
    free(stepY);

    return 0;
}

