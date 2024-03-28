#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <Allocator.h>


namespace core {

void* alloc_cpu(size_t nbytes) {
  // TODO: only simple allocate
  void* data;
  data = aligned_alloc(kAlignment, nbytes);
  return data;
}

std::unique_ptr<void, DeleterFn> CPUAllocator::allocate(size_t n) {
  auto deleter = [](void* ptr){free (ptr);};
  return std::unique_ptr<void, decltype(deleter)>(alloc_cpu(n), deleter);
}

} // namespace core
