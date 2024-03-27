#pragma once
#include <Device.h>
#include <Storage.h>
#include <optional>
#include <vector>
#include <ranges>
#include <type_traits>

namespace core {

class TensorImpl {
public:
  TensorImpl(Storage&& storage);

  TensorImpl() = delete;
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  int64_t numel() const;

  int64_t size(int64_t dim) const {
    return sizes_.at(dim);
  }

  template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  void setSize(const std::vector<T> sizes) {
    sizes_.resize(sizes.size());
    for (auto&& i : std::views::iota(0, sizes.size())) {
      numel_ *= sizes[i];
      sizes_[i] = sizes[i];
    }
  }

  template<typename... TS, typename = std::enable_if_t<std::is_integral_v<std::common_type_t<TS...>>>>
  void setSize(TS&&... args) {
    setSize({args...});
  }


  template<typename T>
  const T* data() const {
    return static_cast<T*>(storage_.data());
  }

  template<typename T>
  T* mutable_data() {
    return static_cast<T*>(storage_.mutable_data());
  }

private:
  Storage storage_;
  // std::optional<Device> device_;
  int64_t numel_;
  // TODO: take llvm small vector
  std::vector<int64_t> sizes_;
};

} // namespace core