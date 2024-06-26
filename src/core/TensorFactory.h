#pragma once
#include <vector>
#include <Tensor.h>
#include <ScalarType.h>
#include "Library.h"

namespace core {

Tensor empty_cpu(const std::vector<int64_t>& sizes, ScalarType type);
Tensor add_tensor_cpu(const Tensor& lhs, const Tensor& rhs);
Tensor eye_cpu(std::vector<int64_t> sizes, ScalarType type);
Tensor ones_cpu(std::vector<int64_t> sizes, ScalarType type);
Tensor random_cpu(std::vector<int64_t> sizes);
Tensor zeros_cpu(std::vector<int64_t> sizes);
Tensor matmul(const Tensor& a, const Tensor& b);


#include "OpRegistry.inc.h"
} // namespace core
