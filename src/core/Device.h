#pragma once
#include <cstdint>

namespace core {

enum class DeviceType :int8_t {
  CPU = 0,
  CUDA = 1,
  COMPILE_TIME_MAX_DEVICE_NUMS,
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr int COMPILE_TIME_MAX_DEVICE_NUMS = 
  static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_NUMS);

class Device {
public:
  Device() : type_(kCPU) {}
  Device(DeviceType type) : type_(type) {}

  bool is_cpu() const noexcept {
    return type_ == kCPU;
  }

  bool is_cuda() const noexcept {
    return type_ == kCUDA;
  }

  DeviceType type() const noexcept {
    return type_;
  }

private:
  DeviceType type_;
  // int index_; // To support multiple devices
};

} // namespace core
