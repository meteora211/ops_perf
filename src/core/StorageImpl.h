#pragma once

#include <memory>
#include <Allocator.h>
#include <Device.h>

namespace core {

class StorageImpl {
public:
  StorageImpl(size_t sizes,
              std::unique_ptr<void, DeleterFn> ptr,
              std::unique_ptr<Allocator> allocator);

  StorageImpl(size_t sizes,
              std::unique_ptr<Allocator> allocator);

  StorageImpl() = delete;
  StorageImpl(StorageImpl&& rhs) = delete;
  StorageImpl(const StorageImpl& rhs) = delete;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl& operator=(StorageImpl&&) = delete;

  const void* data() const {
    return ptr_.get();
  }

  void* mutable_data() {
    return ptr_.get();
  }

private:
  std::unique_ptr<void, DeleterFn> ptr_;
  std::unique_ptr<Allocator> allocator_;
  Device device_;
};

} // namespace core
