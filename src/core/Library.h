#pragma once

#include <array>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "Schema.h"
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
class Library {};

// template<typename FuncType, typename ReturnType, typename ParamLists>
// class Callable;

template <typename FuncType, typename ReturnType, typename... Params>
class Callable {
public:
  explicit Callable(FuncType &&func) : func_(std::forward<FuncType>(func)) {}
  ~Callable() = default;
  decltype(auto) operator()(Params &&...args) {
    return func_(std::forward<Params>(args)...);
  }

private:
  FuncType func_;
};

template <typename FuncType>
using CallableWrapper = Callable<
    FuncType, typename traits::infer_function_traits_t<FuncType>::return_type,
    typename traits::infer_function_traits_t<FuncType>::parameter_types>;

// TODO: Callable wrapper for a kernel function
struct KernelFunction {
  // template<typename Functor>
  // KernelFunction(Functor *f) : functor_(f) {};
  KernelFunction() = default;
  ~KernelFunction() = default;
  KernelFunction(void *functor) : functor_(functor){};
  // KernelFunction& operator=(const KernelFunction& rhs) {
  //   functor_ = rhs.functor_;
  //   return *this;
  // }
  // KernelFunction& operator=(KernelFunction&& rhs) {
  //   functor_ = rhs.functor_;
  //   return *this;
  // }

  template <typename Return, typename... Args>
  Return call(Args &&...args) const {
    // TODO: check nullptr
    using Signature = Return(Args...);
    Signature *sig = reinterpret_cast<Signature *>(functor_);
    return (*sig)(std::forward<Args>(args)...);
  }

  void *functor_;
};

class Operator {
public:
  Operator(Schema &&schema) : schema_(std::move(schema)) {}
  Operator(const Schema &schema) : schema_(schema) {}
  virtual ~Operator() = default;

  virtual std::string name() const { return schema_.name(); }
  virtual const Schema &schema() const { return schema_; }

  virtual bool registerKernel(DispatchKey key, void *kernel) {
    kernelLookupTable_[static_cast<int>(key)] = KernelFunction{kernel};
    return true;
  }

protected:
  Schema schema_;
  // std::string schema_;
  std::array<KernelFunction, num_backends> kernelLookupTable_;
};

template <typename Func> class TypedOperator : public Operator {};

template <typename ReturnType, typename... Params>
class TypedOperator<ReturnType(Params...)> : public Operator {
public:
  TypedOperator(Schema &&name) : Operator(std::move(name)) {}
  TypedOperator(const Schema &name) : Operator(name) {}

  // TODO: dispatch to actual kernel with dispatchkey
  ReturnType call(Params &&...args) const {
    // TODO: return type and universe reference
    auto key = getDispatchKey(std::forward<Params>(args)...);
    return kernelLookupTable_[static_cast<int>(key)]
        .template call<ReturnType, Params...>(std::forward<Params>(args)...);
  }

  template <typename... Args> DispatchKey getDispatchKey(Args &&...args) const {
    // TODO: hardcoded for test
    return DispatchKey::CPU;
  }

  // bool registerKernel(DispatchKey key, KernelFunction kernel) {
  //   kernelLookupTable_[static_cast<int>(key)] = kernel;
  //   return true;
  // }
};

// TODO: Operator registry for different backend
class OperatorRegistry {
public:
  OperatorRegistry() = default;
  ~OperatorRegistry() = default;

  // template <typename Func> Operator registerOperator(Schema &&scheme) {
  template <typename Func>
  Operator &registerOperator(const std::string &name) {
    if (operatorLookupTable_.find(name) != operatorLookupTable_.end()) {
      return operatorLookupTable_.at(name);
    }
    TypedOperator<Func> op(name);
    operatorLookupTable_.emplace(op.schema(), op);
    return operatorLookupTable_.at(name);
  }

  const Operator &getOperator(const Schema &name) {
    auto it = operatorLookupTable_.find(name);
    if (it == operatorLookupTable_.end()) {
      throw std::runtime_error("No operator with name " + name.name());
    }
    return it->second;
  }

private:
  std::unordered_map<Schema, Operator> operatorLookupTable_;
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
// Operator getOperator(const std::string &name);
const Operator &getOperator(const Schema &name);

template <typename Func>
const TypedOperator<Func> &getTypedOperator(const Schema &name) {
  const Operator &op = getOperator(name);
  return *static_cast<const TypedOperator<Func> *>(&op);
}

OperatorRegistry &operatorRegistry();

// REGISTER_OP(add, Tensor(const Tensor&, const Tensor&), CPU, add_cpu_impl)
#define REGISTER_OP(name, sig, backend, kernel)                                \
  struct name##_schema {                                                       \
    using schema = sig;                                                        \
    using ptr_schema = schema *;                                               \
  };                                                                           \
  auto operator##_backend = operatorRegistry()                                 \
      .registerOperator<sig>(#name)                                            \
      .registerKernel(DispatchKey::backend,                                    \
                      reinterpret_cast<void *>(&kernel));

// static TypedOperator<sig> create_##name##_##backend##_operator() {}
