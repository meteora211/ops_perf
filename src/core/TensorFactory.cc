#include <Tensor.h>
#include <ScalarType.h>
#include <TensorFactory.h>
#include <utils.h>

namespace core {

Tensor empty_cpu(std::vector<int64_t> sizes, ScalarType type) {
  uint64_t nbytes = 1;
  bool overflow = safe_calculate_nbytes(sizes, &nbytes, type.itemsize());
  auto storage_impl = std::make_shared<StorageImpl>(nbytes, CPUAllocator());
  auto tensor_impl = std::make_shared<TensorImpl>(storage_impl);
  Tensor t(tensor_impl);
  return t;
}

} // namespace core
