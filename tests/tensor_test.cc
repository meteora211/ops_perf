#include <gtest/gtest.h>

#include "Tensor.h"
#include "TensorFactory.h"

using namespace core;

TEST(TestTensor, TestCreate) {
  Tensor t = empty_cpu({2,3,4}, ScalarType::Float);
  EXPECT_TRUE(t.numel() == 24);
}
