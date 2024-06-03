#pragma once

#include <array>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "traits.h"
#include "Schema.h"

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
  KernelFunction(void *functor) : functor_(functor){};
  ~KernelFunction() = default;

  template <typename Return, typename... Args> Return call(Args &&...args) {
    // TODO: check nullptr
    using Signature = Return(Args...);
    Signature *sig = reinterpret_cast<Signature *>(functor_);
    return (*sig)(std::forward<Args>(args)...);
  }

  void *functor_;
};


class Operator {
public:
  explicit Operator(Schema &&schema) : schema_(std::move(schema)) {}
  explicit Operator(const Schema &schema) : schema_(schema) {}
  virtual ~Operator() = default;

  std::string name() const { return schema_.name(); }
  const Schema& schema() const { return schema_; }

  bool registerKernel(DispatchKey key, void *kernel) {
    kernelLookupTable_[static_cast<int>(key)] = kernel;
    return true;
  }

protected:
  Schema schema_;
  std::array<KernelFunction, num_backends> kernelLookupTable_;
};

template <typename Func> class TypedOperator : public Operator {};

template <typename ReturnType, typename... Params>
class TypedOperator<ReturnType(Params...)> : public Operator {
public:
  explicit TypedOperator(Schema &&name) : Operator(std::move(name)) {}
  explicit TypedOperator(const Schema &name) : Operator(name) {}

  // TODO: dispatch to actual kernel with dispatchkey
  ReturnType call(Params &&...args) {
    // TODO: return type and universe reference
    auto key = getDispatchKey(std::forward<Params>(args)...);
    return kernelLookupTable_[key].template call<ReturnType, Params...>(
        std::forward<Params>(args)...);
  }

  template <typename... Args> DispatchKey getDispatchKey(Args &&...args) const {
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

  // Operator &registerOperator(Operator &op) {
  //   // TODO: register operator
  //   operatorLookupTable_.emplace(op.name(), op);
  //   return op;
  // }

  template <typename Func> Operator registerOperator(Schema &&scheme) {
    // TODO: register operator
    TypedOperator<Func> op(scheme);
    operatorLookupTable_.emplace(op.schema(), op);
    return op;
  }

  Operator getOperator(const Schema& name) {
    // TODO: get operator
    auto it = operatorLookupTable_.find(name);
    if (it == operatorLookupTable_.end()) {
      return Operator(name);
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
Operator getOperator(const Schema &name);

OperatorRegistry &operatorRegistry();

// REGISTER_OP("add", CPU, add_cpu_impl)
#define REGISTER_OP(name, sig, backend, kernel)                                \
  struct name##_schema {                                                       \
    using schema = sig;                                                        \
    using ptr_schema = schema *;                                               \
  };                                                                           \
  auto operator##_backend = operatorRegistry()                               \
      .registerOperator<sig>(Schema(#name, #sig))                          \
      .registerKernel(DispatchKey::backend,                                    \
                      reinterpret_cast<void *>(&kernel));
  // static TypedOperator<sig> create_##name##_##backend##_operator() {}       \

// struct TORCH_API {name} {{
//   using schema = {sig.type()};
//   using ptr_schema = schema*;
//   // See Note [static constexpr char* members for windows NVCC]
//   STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")
//   STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name,
//   "{f.func.name.overload_name}")
//   STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str,
//   {cpp_string(str(f.func))}) static {sig.defn(name="call",
//   is_redispatching_fn=False)}; static {sig.defn(name="redispatch",
//   is_redispatching_fn=True)};
// }};"""
//
//         elif self.target is Target.DEFINITION:
//             defns = f"""
// STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name,
// "aten::{f.func.name.name}") STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name},
// overload_name, "{f.func.name.overload_name}")
// STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str,
// {cpp_string(str(f.func))})
//
// // aten::{f.func}
// static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema>
// create_{name}_typed_handle() {{
//   return c10::Dispatcher::singleton()
//       .findSchemaOrThrow({name}::name, {name}::overload_name)
//       .typed<{name}::schema>();
// }}
