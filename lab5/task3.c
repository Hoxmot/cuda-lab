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

    Matrix M1, M2, ResCpu;
    
    M1.stride = M1.height = M1.width = LEN;
    M1.elements = (double*)calloc(LEN * LEN, sizeof(double));

    M2.stride = M2.width = M2.height = LEN; 
    M2.elements = (double*)calloc(LEN * LEN, sizeof(double));

    for (int i = 0; i < LEN; i++) {
        for (int k = 0; k < LEN; k++) {
            M1.elements[i * LEN + k] = (i + k) + 7;
            M2.elements[i * LEN + k] = (i + k) + 3;
        }
    }

    ResCpu.height = ResCpu.stride = ResCpu.width = LEN;
    ResCpu.elements = (double*)calloc(LEN * LEN, sizeof(double));

    matrix_mul(M1, M2, ResCpu);

    free(M1.elements);
    free(M2.elements);
    free(ResCpu.elements);

    return 0;

}