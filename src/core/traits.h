#pragma once
#include <cstdint>
#include <type_traits>

namespace traits {

template <typename Functor>
struct FunctionTraits;

template <typename Result, typename... Params>
struct FunctionTraits<Result(Params...)> {
  using result_type = Result;
  using parameter_types = std::tuple<Params...>;
  static constexpr std::size_t parameter_count = sizeof...(Params);
};

template<typename Functor>
struct infer_function_traits {
  using type = typename FunctionTraits<decltype(&Functor::operator())>::type;
};

template<typename Result, typename... Params>
struct infer_function_traits<Result(Params...)> {
  using type = FunctionTraits<Result(Params...)>;
};

template<typename Result, typename... Params>
struct infer_function_traits<Result(*)(Params...)> {
  using type = FunctionTraits<Result(Params...)>;
};

template<typename T>
using infer_function_traits_t = typename infer_function_traits<T>::type;

} // namespace traits
