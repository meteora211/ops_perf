#include <Tensor.h>
#include <TensorFactory.h>

namespace core {

Tensor empty_cpu(std::vector<int64_t> sizes) {
  auto storage_impl = std::make_shared<Storage>();
  auto tensor_impl = 
  Tensor t;
  return t;
}

} // namespace core
