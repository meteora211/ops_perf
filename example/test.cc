#include <Tensor.h>
#include <TensorFactory.h>
using namespace core;


int main() {
  Tensor t = empty_cpu({2,3,4}, ScalarType::Float);
}
