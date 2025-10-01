#pragma once

#include <string>
#include <unordered_map>
#include "../tensor.hpp"

std::unordered_map<std::string, Tensor> load_onnx_initializers(const std::string& onnx_path);