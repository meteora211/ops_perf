#pragma once
#include <optional>
#include <vector>
#include <ranges>
#include <type_traits>
#include <limits>
#include <iostream>
#include "ScalarType.h"
#include "Device.h"
#include "Storage.h"
#include "utils.h"

namespace core {

enum class MemoryFormat : int8_t {
  Contiguous,
  Undefine,
};

class TensorImpl {
public:
  TensorImpl(Storage&& storage, ScalarType dtype);

  TensorImpl() = delete;
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  int64_t numel() const;

  int64_t dim() const {
    return sizes_.size();
  }

  int64_t size(int64_t dim) const {
    return sizes_.at(dim);
  }

  std::vector<int64_t> sizes() const {
    return sizes_;
  }

  std::vector<int64_t> strides() const {
    return strides_;
  }

  void refresh_strides() {
    auto dim_ = dim();
    if (dim_ > 0) {
      int64_t stride = 1;
      bool overflowed = false;
      strides_[dim_ - 1] =  1;
      for (int i = dim_ - 2; i >= 0; --i) {
        overflowed |= safe_mul(stride, sizes_[i + 1], &stride);
        strides_[i] =  stride;
      }
      // TODO: add logging and assert
      if (overflowed) {
        std::cout << "overflowed" << std::endl;
      }
    }
  }

  int64_t calculate_numel() const {
    uint64_t u_numel{0};
    auto overflowed = safe_calculate_numel(sizes_, &u_numel);

    constexpr auto numel_max = std::min(
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

    overflowed |= (u_numel > numel_max);
    // TODO: add logging and assert
    if (overflowed) {
      std::cout << "overflowed" << std::endl;
    }
    return static_cast<int64_t>(u_numel);
  }

  void refresh_numel() {
    numel_ = calculate_numel();
  }

  template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  void set_contignuous_sizes(const std::vector<T>& sizes) {
    sizes_.resize(sizes.size());
    strides_.resize(sizes.size());
    for (int i : std::views::iota(0, static_cast<int>(sizes.size()))) {
      numel_ *= sizes[i];
      sizes_[i] = sizes[i];
    }
    refresh_strides();
  }

  template<typename... TS, typename = std::enable_if_t<std::is_integral_v<std::common_type_t<TS...>>>>
  void set_contignuous_sizes(TS&&... args) {
    set_contignuous_sizes({args...});
  }


  template<typename T>
  const T* data() const {
    return static_cast<T*>(storage_.data());
  }

  template<typename T>
  T* mutable_data() {
    return static_cast<T*>(storage_.mutable_data());
  }

  ScalarType dtype() const {
    return data_type_;
  }

private:
  Storage storage_;
  // std::optional<Device> device_;
  int64_t numel_;
  ScalarType data_type_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  MemoryFormat memory_format_;
};

} // namespace core
