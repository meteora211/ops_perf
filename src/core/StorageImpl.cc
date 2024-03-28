#include <Device.h>
#include <StorageImpl.h>

namespace core {

StorageImpl::StorageImpl(
  size_t sizes,
  std::unique_ptr<void, DeleterFn> ptr,
  std::unique_ptr<Allocator> allocator)
  : ptr_(std::move(ptr)), allocator_(std::move(allocator))
{
  // TODO: create device_ according to allocator
  device_ = Device(kCPU);
}

StorageImpl::StorageImpl(
  size_t sizes,
  std::unique_ptr<Allocator> allocator)
  : StorageImpl(sizes,
                allocator->allocate(sizes),
                std::move(allocator)) {

}

} // namespace core
