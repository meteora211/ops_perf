#pragma once

#include <unordered_map>
#include <string>
#include <list>
#include <vector>
#include <array>
#include "traits.h"

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

// template<typename FuncType, typename ReturnType, typename ParamLists>
// class Callable;

template<typename FuncType, typename ReturnType, typename... Params>
class Callable {
public:
  explicit Callable(FuncType&& func) : func_(std::forward<FuncType>(func)) {}
  ~Callable() = default;
  decltype(auto) operator()(Params&&... args) {
    return func_(std::forward<Params>(args)...);
  }
private:
  FuncType func_;
};

template<typename FuncType>
using CallableWrapper = Callable<
  FuncType,
  typename traits::infer_function_traits_t<FuncType>::return_type,
  typename traits::infer_function_traits_t<FuncType>::parameter_types
>;


// TODO: Callable wrapper for a kernel function
struct KernelFunction {
  // template<typename Functor>
  // KernelFunction(Functor *f) : functor_(f) {};
  KernelFunction() = default;
  KernelFunction(void* functor) : functor_(functor) {};
  ~KernelFunction() = default;

  template<typename Return, typename... Args>
  Return call(Args&&... args) {
    // TODO: check nullptr
    using Signature = Return(Args...);
    Signature* sig = reinterpret_cast<Signature*>(functor_);
    return (*sig)(std::forward<Args>(args)...);
  }

  void* functor_;
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
  Operator(std::string name) : schema_(name) {}
  virtual ~Operator() = default;

  std::string name() const {
    return schema_.name();
  }


  bool registerKernel(DispatchKey key, void* kernel) {
    kernelLookupTable_[static_cast<int>(key)] = kernel;
    return true;
  }

protected:
  Schema schema_;
  std::array<KernelFunction, num_backends> kernelLookupTable_;
};

template<typename Func>
class TypedOperator : public Operator {
};

template<typename ReturnType, typename... Params>
class TypedOperator<ReturnType(Params...)> : public Operator {
public:
  explicit TypedOperator(std::string name) : Operator(name) {}

  // TODO: dispatch to actual kernel with dispatchkey
  ReturnType call(Params&&... args) {
    // TODO: return type and universe reference
    auto key = getDispatchKey(std::forward<Params>(args)...);
    return kernelLookupTable_[key].template call<ReturnType, Params...>(std::forward<Params>(args)...);
  }

  template<typename... Args>
  DispatchKey getDispatchKey(Args&&... args) const {
    // TODO: hardcoded for test
    return DispatchKey::CPU;
  }

  bool registerKernel(DispatchKey key, KernelFunction kernel) {
    kernelLookupTable_[static_cast<int>(key)] = kernel;
    return true;
  }
};


// TODO: Operator registry for different backend
class OperatorRegistry {
public:
  OperatorRegistry() = default;
  ~OperatorRegistry() = default;

  Operator& registerOperator(Operator& op) {
    // TODO: register operator
    // operatorLookupTable_[op.name()] = op;
    operatorLookupTable_.emplace(op.name(), op);
    return op;
  }

  Operator registerOperator(Schema&& scheme) {
    // TODO: register operator
    Operator op(scheme.name());
    // operatorLookupTable_[op.name()] = op;
    operatorLookupTable_.emplace(op.name(), op);
    return op;
  }

  Operator getOperator(const std::string &name) {
    // TODO: get operator
    auto it = operatorLookupTable_.find(name);
    if (it == operatorLookupTable_.end()) {
      return Operator(name);
    }
    return it->second;
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
  bool operator##_backend = operatorRegistry() \
    .registerOperator(Schema(schema))  \
    .registerKernel(DispatchKey::backend, reinterpret_cast<void*>(&kernel));
