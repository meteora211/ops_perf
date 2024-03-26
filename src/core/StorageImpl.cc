#include <Device.h>
#include <StorageImpl.h>

namespace core {

StorageImpl::StorageImpl(size_t sizes,
                         std::unique_ptr<Allocator> allocator) : allocator_(std::move(allocator)) {
  ptr_ = allocator_->allocate(sizes);
  // TODO: create device_ according to allocator
  device_ = Device(kCPU);
}

} // namespace core
