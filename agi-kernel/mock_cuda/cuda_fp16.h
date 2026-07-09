#pragma once

#ifndef MOCK_CUDA_FP16_H
#define MOCK_CUDA_FP16_H

#include <cstdint>

struct __half {
    uint16_t x;
    __half() : x(0) {}
    __half(float f) {
        // Mock conversion: just 0
        x = 0;
    }
    operator float() const {
        return 0.0f;
    }
};

#endif // MOCK_CUDA_FP16_H
