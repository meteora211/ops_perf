#pragma once

#include <memory>
#include <Allocator.h>
#include <Device.h>

namespace core {

class StorageImpl {
public:
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& rhs) {
    ptr_ = std::move(rhs.ptr_);
    allocator_ = std::move(rhs.allocator_);
    device_ = std::move(rhs.device_);
  }
  StorageImpl(size_t sizes,
              std::unique_ptr<Allocator> allocator);

  const void* data() const {
    return ptr_.get();
  }

  void* mutable_data() {
    return ptr_.get();
  }

private:
  std::unique_ptr<void*> ptr_;
  std::unique_ptr<Allocator> allocator_;
  Device device_;
};

} // namespace core
