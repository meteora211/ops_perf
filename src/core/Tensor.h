#pragma once

#include <memory>
#include <exception>
#include <stdexcept>
#include <TensorImpl.h>

namespace core {


class Tensor {
public:
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    if (impl_ == nullptr) {
      throw std::runtime_error("TensorImpl can not be nullptr.");
    }
  }

  ~Tensor()=default;

  Tensor& operator=(const Tensor& rhs) {
    impl_ = rhs.getIntrusivePtr();
    return *this;
  }

  std::shared_ptr<TensorImpl> getIntrusivePtr() const {
    return impl_;
  }

  TensorImpl* unsafeGetTensorImpl() {
    return impl_.get();
  }

  bool defined() const {
    return impl_ != nullptr;
  }

  void reset() {
    impl_.reset();
  }

  int64_t size(int64_t dim) const {
    return impl_->size(dim);
  }

  // TODO: only valid for supported datatype T
  // TODO: when T contains const
  template<typename T>
  const T* data() const {
    return impl_->data<T>();
  }

  template<typename T>
  T* mutable_data() {
    return impl_->mutable_data<T>();
  }


private:
  std::shared_ptr<TensorImpl> impl_;
};

} // namespace core
