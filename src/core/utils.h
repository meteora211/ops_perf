#pragma once
#include <cstdint>
#include <iostream>
#include <vector>

namespace core {

inline bool safe_mul(uint64_t a, uint64_t b, uint64_t *out) {
  // TODO: check the c10::llvm implement
  volatile uint64_t c = a * b;
  if (a == 0 || b == 0)
    return false;
  *out = a * b;
  return !(a == c / b);
}

inline bool safe_mul(int64_t a, int64_t b, int64_t *out) {
  volatile int64_t c = a * b;
  *out = c;
  if (a == 0 || b == 0)
    return false;
  return !(a == c / b);
}

template <typename It>
bool safe_calculate_numel(It begin, It end, uint64_t *numel) {
  bool overflow = false;
  uint64_t acc = 1;
  for (It iter = begin; iter != end; ++iter) {
    overflow |= safe_mul(*iter, acc, &acc);
  }
  *numel = acc;
  return overflow;
}

template <typename Container>
bool safe_calculate_numel(const Container &sizes, uint64_t *numel) {
  return safe_calculate_numel(sizes.begin(), sizes.end(), numel);
}

template <typename Container>
bool safe_calculate_nbytes(const Container &sizes, uint64_t item_size,
                           uint64_t *numel) {
  bool overflowed = false;
  overflowed |= safe_calculate_numel(sizes.begin(), sizes.end(), numel);
  overflowed |= safe_mul(*numel, item_size, numel);
  return overflowed;
}

} // namespace core
