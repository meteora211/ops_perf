#pragma once
#include <vector>
#include <Tensor.h>

namespace core {

Tensor empty_cpu(std::vector<int64_t> sizes);
Tensor eye_cpu(std::vector<int64_t> sizes);
Tensor ones_cpu(std::vector<int64_t> sizes);
Tensor random_cpu(std::vector<int64_t> sizes);
Tensor zeros_cpu(std::vector<int64_t> sizes);

} // namespace core
