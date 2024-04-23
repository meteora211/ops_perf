#pragma once

#include <memory>
#include "Allocator.h"
#include "Device.h"
#include "StorageImpl.h"

namespace core {

class Storage {
public:
  Storage() = default;
  Storage(std::shared_ptr<StorageImpl> rhs) : impl_(std::move(rhs)) {}

  const void* data() const {
    return impl_->data();
  }

  void* mutable_data() {
    return impl_->mutable_data();
  }

private:
  std::shared_ptr<StorageImpl> impl_;
};

} // namespace core
