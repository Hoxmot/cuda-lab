/*
  modified from source:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
*/

#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 32

#ifndef LEN
#error LEN is not defined
#endif

typedef struct {
    int width;
    int height;
    int stride;
    double *elements;
} Matrix;

void matrix_mul(const Matrix A, const Matrix B, Matrix C) {
    for (int i = 0; i < LEN; i++) {
        for (int j = 0; j < LEN; j++) {
            C.elements[i * C.stride + j] = 0;
            for (int k = 0; k < LEN; k++) {
                C.elements[i * C.stride + j] += A.elements[i * A.stride + k] + B.elements[k * B.stride + j];
            }
        }
    }
}

int main() {

    Matrix M1_cpu, M2_cpu, ResCpu;
    
    M1_cpu.stride = M1_cpu.height = M1_cpu.width = LEN;
    M1_cpu.elements = (double*)calloc(LEN * LEN, sizeof(double));

    M2_cpu.stride = M2_cpu.width = M2_cpu.height = LEN; 
    M2_cpu.elements = (double*)calloc(LEN * LEN, sizeof(double));

    for (int i = 0; i < LEN; i++) {
        for (int k = 0; k < LEN; k++) {
            M1_cpu.elements[i * LEN + k] = (i + k) + 7;
            M2_cpu.elements[i * LEN + k] = (i + k) + 3;
        }
    }

    ResCpu.height = ResCpu.stride = ResCpu.width = LEN;
    ResCpu.elements = (double*)calloc(LEN * LEN, sizeof(double));

    matrix_mul_cpu(M1_cpu, M2_cpu, ResCpu);

    free(M1_cpu.elements);
    free(M2_cpu.elements);
    free(ResCpu.elements);

    return 0;

}