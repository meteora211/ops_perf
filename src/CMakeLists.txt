add_subdirectory(utils)
# add_subdirectory(benchmark)
# add_subdirectory(ops)

file(GLOB CORE_HEADERS "core/*.h")
file(GLOB CORE_SRCS "core/*.cc")
add_library(core SHARED ${CORE_SRCS})
target_include_directories(core PUBLIC "core")

file(GLOB CPU_OPS "cpu/*.cc")
file(GLOB CUDA_OPS "gpu/*.cu")

set(CPU_OPS_INCLUDE)
set(GPU_OPS_INCLUDE)
set(OPS_UTILS_INCLUDE)

list(APPEND CPU_OPS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cpu)
list(APPEND GPU_OPS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
list(APPEND OPS_UTILS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

add_library(ops SHARED ${CPU_OPS} ${CUDA_OPS})
# target_link_libraries(ops fmt ${CUDA_CUBLAS_LIBRARIES} utils)
target_link_libraries(ops PRIVATE fmt cudnn cublas utils)

target_include_directories(ops PUBLIC "utils")
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
# "$<INSTALL_INTERFACE:${OPS_INSTALL_INCLUDE_DIR}>")

set_target_properties(ops
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS ops LIBRARY DESTINATION ${OPS_INSTALL_LIB_DIR})

set(CPU_OPS_INCLUDE ${CPU_OPS_INCLUDE} PARENT_SCOPE)
set(GPU_OPS_INCLUDE ${GPU_OPS_INCLUDE} PARENT_SCOPE)
set(OPS_UTILS_INCLUDE ${OPS_UTILS_INCLUDE} PARENT_SCOPE)
