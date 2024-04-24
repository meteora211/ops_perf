#include <gtest/gtest.h>
#include <iostream>

#include "Tensor.h"
#include "TensorFactory.h"

using namespace core;

TEST(TestTensor, TestCreate) {
  Tensor t = empty_cpu({2,3,4}, ScalarType::Float);
  std::cout << t.numel() << std::endl;
  EXPECT_TRUE(t.numel() == 24);
}

TEST(TestTensor, ONES) {
  Tensor t = ones_cpu({2,3,4}, ScalarType::Float);
  std::cout << t.numel() << std::endl;
  EXPECT_TRUE(t.numel() == 24);
  const float* data = t.data<const float>();
  for (int i = 0; i < 24; ++i) {
    EXPECT_TRUE(data[i] == 1.f);
  }
}
