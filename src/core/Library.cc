#include "Library.h"


OperatorRegistry &operatorRegistry() {
  static OperatorRegistry operatorRegistry_;
  return operatorRegistry_;
}

Operator getOperator(const std::string &name) {
  return operatorRegistry().getOperator(name);
}


