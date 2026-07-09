#pragma once

#ifndef MOCK_CUDA_RUNTIME_H
#define MOCK_CUDA_RUNTIME_H

#include <cstddef>
#include <cstdio>
#include <cstring>

#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif

#define cudaSuccess 0
typedef int cudaError_t;

typedef void* cudaStream_t;

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

template<typename T>
inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    *devPtr = (T*)new char[size];
    return cudaSuccess;
}

template<typename T>
inline cudaError_t cudaFree(T* devPtr) {
    delete[] (char*)devPtr;
    return cudaSuccess;
}

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0) {
    std::memset(devPtr, value, count);
    return cudaSuccess;
}

#endif // MOCK_CUDA_RUNTIME_H
