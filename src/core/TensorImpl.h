#pragma once
#include <Device.h>
#include <Storage.h>
#include <optional>

namespace core {

class TensorImpl {
public:
  TensorImpl(Storage&& storage);

  TensorImpl() = delete;
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  int64_t numel() const;

  int64_t size(int64_t dim) const {
    // return storage_.size(dim);
    return 1;
  }

  template<typename T>
  const T* data() const {
    return static_cast<T*>(storage_.data());
  }

  template<typename T>
  T* mutable_data() {
    return static_cast<T*>(storage_.mutable_data());
  }

private:
  Storage storage_;
  // std::optional<Device> device_;
  int64_t numel_;
};

} // namespace core
