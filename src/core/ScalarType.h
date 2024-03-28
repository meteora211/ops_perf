#pragma once
#include <cstdint>
#include <stdexcept>

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

#define SCALAR_ITEM_SIZE(BUILTIN, SCALARTYPE) \
template<> \
struct ScalarItemSize<ScalarType::TYPE> { \
static constexpr std::size_t value = sizeof(BUILTIN); \
};

// ALL_SCALAR_TYPES(SCALAR_TYPE_TRAITS)
#undef SCALAR_ITEM_SIZE

constexpr uint16_t NumScalarTypes = static_cast<uint16_t>(ScalarType::NumTypes);

inline std::size_t scalar_type_size(ScalarType type) {
#define TYPE_CASES(builtin, name) \
  case ScalarType::name: \
    return sizeof(builtin);

  switch (type) {
    ALL_SCALAR_TYPES(TYPE_CASES)
    default:
    throw std::runtime_error("Unkown data type");
  }
#undef TYPE_CASES
}
