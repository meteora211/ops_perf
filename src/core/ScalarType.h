#pragma once
#include <cstdint>

#define ALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(float, Float) /* 5 */                                \
  _(double, Double) /* 6 */ 

enum class ScalarType : int8_t {
#define GET_2ND_NAME(_1, TYPE) TYPE,
  ALL_SCALAR_TYPES(GET_2ND_NAME)
#undef GET_2ND_NAME 
  Undefine,
  NumTypes,
};

template<typename T>
struct ScalarTypeTraits;

#define SCALAR_TYPE_TRAITS(TYPE, _) \
template<> \
struct ScalarTypeTraits<TYPE> { \
static constexpr itemsize = sizeof(TYPE); \
};

ALL_SCALAR_TYPES(SCALAR_TYPE_TRAITS)
#undef SCALAR_TYPE_TRAITS

constexpr uint16_t NumScalarTypes = static_cast<uint16_t>(ScalarType::NumTypes);
