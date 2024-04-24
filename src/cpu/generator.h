#pragma once
#include <stdint.h>
#include <vector>
#include <iostream>

template<typename T>
void full_kernel(T* data, T value, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = value;
    std::cout << i << " " << data[i] << std::endl;
  }
}

