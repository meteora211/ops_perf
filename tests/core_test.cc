#include <gtest/gtest.h>
#include <vector>
#include <iostream>

#include "Library.h"
#include "TensorFactory.h"

using namespace core;

TEST(TestCore, TestRegister) {
  Tensor a = ones_cpu({2,3}, ScalarType::Float);
  Tensor b = ones_cpu({2,3}, ScalarType::Float);
  const auto& add = getTypedOperator<Tensor(const Tensor&, const Tensor&)>(Schema("add"));
  auto c = add.call(a,b);
  EXPECT_TRUE(c.numel() == 6);
  const float* data = c.data<const float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(data[i] == 2.f);
  }
}
