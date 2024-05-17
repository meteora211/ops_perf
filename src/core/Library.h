#pragma once

#include <unordered_map>
#include <string>
#include <list>
#include <vector>

enum class DispatchKey : uint16_t {
  CPU,
  GPU,
  Lazy,
  EndOfKey,
};

constexpr uint8_t num_backends = static_cast<uint8_t>(DispatchKey::EndOfKey);

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
  template<typename... Args>
  void operator()(Args&&... args) {}
};

// TODO: use function schema instead of string
class Schema {
public:
  Schema() = default;
  ~Schema() = default;

  Schema(std::string name) : name_(name) {}

  std::string name() const {
    return name_;
  }

private:
  std::string name_;
};

class Operator {
public:
  Operator() = default;
  ~Operator() = default;
  Operator(std::string name) : schema_(name) {}

  std::string name() const {
    return schema_.name();
  }

  // TODO: dispatch to actual kernel with dispatchkey
  template<typename... Args>
  void call(Args&&... args) {
    // TODO: return type
    auto key = getDispatchKey(std::forward<Args>(args)...);
    return kernelLookupTable_[key](args...);
  }

  template<typename... Args>
  DispatchKey getDispatchKey(Args&&... args) const {
    return DispatchKey::CPU;
  }

  void registerKernel(DispatchKey key, KernelFunction kernel) {
    kernelLookupTable_[key] = kernel;
  }
  

private:
  std::array<KernelFunction, num_backends> kernelLookupTable_;
  Schema schema_;
};


// TODO: Operator registry for different backend
class OperatorRegistry {
public:
  OperatorRegistry() = default;
  ~OperatorRegistry() = default;

  Operator registerOperator(Operator op) {
    // TODO: register operator
    operatorLookupTable_[op.name()] = op;
    return op;
  }

  Operator getOperator(const std::string &name) {
    // TODO: get operator
    if (operatorLookupTable_.find(name) == operatorLookupTable_.end()) {
      return Operator();
    }
    return operatorLookupTable_[name];
  }

private:
  // std::vector<KernelFunction> kernelLookupTable_;
  std::unordered_map<std::string, Operator> operatorLookupTable_;
};

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

// TODO: using operator schema
Operator getOperator(const std::string &name);

OperatorRegistry &operatorRegistry();


// REGISTER_OP("add", CPU, add_cpu_impl)
#define REGISTER_OP(schema, backend, kernel)                                  \
  bool operator##_backend = operatorRegistry().registerOperator(schema).registerKernel(backend, kernel);
