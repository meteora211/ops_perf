#include <gtest/gtest.h>
#include <vector>
#include <iostream>

#include "Tensor.h"
#include "TensorFactory.h"

using namespace core;

TEST(TestTensor, TestCreate) {
  Tensor t = empty_cpu({2,3,4}, ScalarType::Float);
  EXPECT_TRUE(t.numel() == 24);
  bool check_size = t.sizes() == std::vector<int64_t>{2, 3, 4};
  EXPECT_TRUE(check_size);
  bool check_stride = t.strides() == std::vector<int64_t>{12, 4, 1};
  EXPECT_TRUE(check_stride);
}

TEST(TestTensor, TestOnes) {
  Tensor t = ones_cpu({2,3,4}, ScalarType::Float);
  EXPECT_TRUE(t.numel() == 24);
  const float* data = t.data<const float>();
  for (int i = 0; i < 24; ++i) {
    EXPECT_TRUE(data[i] == 1.f);
  }
}

TEST(TestTensor, TestAdd) {
  Tensor a = ones_cpu({2,3}, ScalarType::Float);
  Tensor b = ones_cpu({2,3}, ScalarType::Float);
  Tensor c = add_tensor_cpu(a, b);
  EXPECT_TRUE(c.numel() == 6);
  const float* data = c.data<const float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(data[i] == 2.f);
  }
}
