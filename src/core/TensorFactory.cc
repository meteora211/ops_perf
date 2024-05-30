#include "TensorFactory.h"
#include "Tensor.h"
#include "utils.h"
#include "Library.h"
#include <iostream>

// TODO: headers for all kernel api
#include "cpu_add.h"
#include "cpu_matmul.h"
#include "generator.h"

namespace core {

Tensor empty_cpu(const std::vector<int64_t> &sizes, ScalarType type) {
  uint64_t nbytes = 1;
  // TODO: add assert logging
  bool overflowed = safe_calculate_nbytes(sizes, scalar_type_size(type), &nbytes);
  auto allocator = std::make_unique<CPUAllocator>();
  auto storage_impl =
      std::make_shared<StorageImpl>(nbytes, std::move(allocator));
  auto tensor_impl = std::make_shared<TensorImpl>(storage_impl, type);
  tensor_impl->set_contignuous_sizes(sizes);
  Tensor t(tensor_impl);
  return t;
}

Tensor add_tensor_cpu(const Tensor &lhs, const Tensor &rhs) {
  const auto &lhs_sizes = lhs.sizes();
  const auto &rhs_sizes = rhs.sizes();
  // TODO: data type check
  // TODO: assume lhs/rhs has same input sizes and add broadcast handling later

  Tensor out = empty_cpu(lhs_sizes, lhs.dtype());
  // TODO: how to automatic detect and fetch data with type?
  // TODO: remove const in data<const float>?
  const auto *lhs_data = lhs.data<const float>();
  const auto *rhs_data = rhs.data<const float>();
  auto *out_data = out.mutable_data<float>();
  // TODO: how to dispatch on different type?
  // TODO: binary ops
  // TODO: how to handle stride, shape and dimentions?
  add_kernel(lhs_data, rhs_data, out_data, lhs.sizes()[0], lhs.sizes()[1]);

  return out;
}

Tensor ones_cpu(std::vector<int64_t> sizes, ScalarType type) {
  Tensor out = empty_cpu(sizes, type);
  float *out_data = out.mutable_data<float>();
  auto numel = out.numel();
  full_kernel(out_data, 1.f, numel);

  return out;
}

REGISTER_OP("add", CPU, add_tensor_cpu)

// template<typename T>
// void matmul2d_cpu(const core::Tensor& lhs, const core::Tensor& rhs,
// core::Tensor& output) {
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
