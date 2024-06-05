#include "Library.h"
#include "Schema.h"


OperatorRegistry &operatorRegistry() {
  static OperatorRegistry operatorRegistry_;
  return operatorRegistry_;
}

const Operator& getOperator(const Schema& name) {
  return operatorRegistry().getOperator(name);
}
