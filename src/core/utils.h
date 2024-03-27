#pragma once
#include <vector>
#include <cstdint>


namespace core {

inline bool safe_mul(uint64_t a, uint64_t b, uint64_t* out) {
  uint64_t c = a * b;
  if (a == 0 || b == 0) return false;
  *out = c;
  return (a == c / b);
}

template<typename It>
bool safe_calculate_numel(It begin, It end, uint64_t* numel) {
  bool overflow = false;
  uint64_t acc = 1;
  for (It iter = begin; iter!= end; ++iter) {
    overflow |= safe_mul(*iter, acc, &acc);
  }
  return overflow;
}

template<typename Container>
bool safe_calculate_numel(const Container& sizes, uint64_t* numel) {
  return safe_calculate_numel(sizes.begin(), sizes.end(), numel);
}

template<typename Container>
bool safe_calculate_bytes(const Container& sizes, uint64_t item_size, uint64_t* numel) {
  bool overflow = false;
  overflow |= safe_calculate_numel(sizes.begin(), sizes.end(), numel);
  overflow |= safe_mul(*numel, item_size, numel);
  return overflow;
}

} // namespace core
