#pragma once

#include <unordered_map>
#include <string>
#include <list>
#include <vector>

// Library (core, custom)
//   Operator (add, sub, ...)
//     Kernel (cpu, gpu, ...)
//
//
// TODO: Library for register custom operator
class Library {
};

// TODO: Callable wrapper for a kernel function
struct KernelFunction {

};

class Schema;

class Operator {
public:
  Operator() = default;
  ~Operator() = default;

private:
  std::vector<KernelFunction> kernelLookupTable_;
  Schema schema_;
};


// TODO: Operator registry for different backend
class OperatorRegistry {
public:
  OperatorRegistry() = default;
  ~OperatorRegistry() = default;

  void registerOperator(Operator op) {
    // TODO: register operator
  }

private:
  // std::vector<KernelFunction> kernelLookupTable_;
  std::unordered_map<std::string, Operator> operatorLookupTable_;
};

// TODO: using operator schema
Operator getOperator(const std::string &name);

// class Dispatcher {
// private:
//   // TODO: use string to identify operator temporarily
//   std::unordered_map<std::string, OperatorHandler> operatorLookupTable_;
// };


// Example usage:
//
// ```
// MET_LIBRARY(myops, m) {
//   // m is a Library; methods on it will define
//   // operators in the myops namespace
//   m.def("add", add_impl);
// }
// ```
// #define MET_LIBRARY(ns, m)
///
/// TORCH_LIBRARY_IMPL(myops, CPU, m) {
///   // m is a torch::Library; methods on it will define
///   // CPU implementations of operators in the myops namespace.
///   // It is NOT valid to call torch::Library::def()
///   // in this context.
///   m.impl("add", add_cpu_impl);
/// }


OperatorRegistry &operatorRegistry() {
  static OperatorRegistry operatorRegistry_;
  return operatorRegistry_;
}


#define REGISTER_OP(schema)                                               \
  bool operator##_entry = operatorRegistry().registerOperator(schema);
