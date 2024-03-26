#pragma once
#include <memory>

namespace core {

void* alloc_cpu(size_t nbytes);

constexpr int kAlignment = 64;

class Allocator {
public:
  virtual ~Allocator() = default;
  // TODO: deleter
  virtual std::unique_ptr<void> allocate(size_t n) = 0;
};

class CPUAllocator : public Allocator {
public:
  CPUAllocator() = default;
  std::unique_ptr<void> allocate(size_t n) override;
};

} // namespace core
