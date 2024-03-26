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

std::unique_ptr<void> CPUAllocator::allocate(size_t n) {
  return std::unique_ptr<void>(alloc_cpu(n));
}

} // namespace core
