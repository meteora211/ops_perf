#include "Tensor.h"
#include "TensorFactory.h"
#include "utils.h"

// TODO: headers for all kernel api
#include "cpu_matmul.h"
#include "cpu_add.h"

namespace core {

Tensor empty_cpu(const std::vector<int64_t>& sizes, ScalarType type) {
  uint64_t nbytes = 1;
  bool overflow = safe_calculate_nbytes(sizes, scalar_type_size(type), &nbytes);
  auto allocator = std::make_unique<CPUAllocator>();
  auto storage_impl = std::make_shared<StorageImpl>(nbytes, std::move(allocator));
  auto tensor_impl = std::make_shared<TensorImpl>(storage_impl);
  Tensor t(tensor_impl);
  return t;
}

Tensor add_tensor_cpu(const Tensor& lhs, const Tensor& rhs) {
  const auto& lhs_sizes = lhs.sizes();
  const auto& rhs_sizes = rhs.sizes();

  Tensor t = empty_cpu(sizes, dtype);
  add_kernel(lhs_data, rhs_data, out_data);

  return t;
}

// template<typename T>
// void matmul2d_cpu(const core::Tensor& lhs, const core::Tensor& rhs, core::Tensor& output) {
//   const T* lhs_ptr = lhs.data<T>();
//   const T* rhs_ptr = rhs.data<T>();
//   T* out_ptr = output.mutable_data<T>();

//   int64_t M = lhs.size(0);
//   int64_t K = lhs.size(1);
//   int64_t N = rhs.size(1);

//   matmul_block(lhs_ptr,
//                rhs_ptr,
//                out_ptr,
//                M,
//                N,
//                K);
// }

// Tensor matmul(const Tensor& a, const Tensor& b) {
//   Tensor c = empty_cpu(sizes);
//   matmul2d_cpu(a, b, c);
//   return c;
// }

} // namespace core
