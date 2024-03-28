#pragma once
#include <memory>

namespace core {

void* alloc_cpu(size_t nbytes);

constexpr int kAlignment = 64;
using DeleterFn = void(*)(void*);

class Allocator {
public:
  virtual ~Allocator() = default;
  // TODO: deleter
  virtual std::unique_ptr<void, DeleterFn> allocate(size_t n) = 0;
};

class CPUAllocator : public Allocator {
public:
  CPUAllocator() = default;
  std::unique_ptr<void, DeleterFn> allocate(size_t n) override;
};

} // namespace core
