#pragma once

#include <vector>
#include <cstddef>

// Forward declaration for now
enum class DType {
    FP32,
    FP16,
    INT8,
    Q4,
};

class Tensor {
public:
    // A dummy implementation for now
    Tensor() = default;
    Tensor(const std::vector<int>& shape, DType dtype) {}
    ~Tensor() {}

    void* raw() { return nullptr; }
    template<typename T> T* data() { return nullptr; }
    size_t byte_size() const { return 0; }
    std::vector<int> shape() const { return {}; }
};