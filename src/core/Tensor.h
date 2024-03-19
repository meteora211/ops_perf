#pragma once

#include <memory>
#include <TensorImpl.h>

class Tensor {
public:
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    if (impl_ == nullptr) {
      throw std::runtime_error("TensorImpl can not be nullptr.")
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
    return impl_;
  }

  void reset() {
    impl_.reset();
  }

private:
  std::shared_ptr<TensorImpl> impl_;
};
