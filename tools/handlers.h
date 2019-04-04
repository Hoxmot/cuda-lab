#ifndef __HANDLERS_H__
#define __HANDLERS_H__

#include <cuda_runtime_api.h>

void handleCudaMalloc(void **var, ssize_t size);

void handleCudaMemcpy(void* dst, const void* src, ssize_t size, cudaMemcpyKind kind);

void handleCudaFree(void* pointer);
#endif