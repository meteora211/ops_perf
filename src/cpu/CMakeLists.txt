file(GLOB CPU_OPS "*.cc")
file(GLOB CUDA_OPS "*.cu")
add_library(ops SHARED ${CPU_OPS} ${CUDA_OPS})
target_link_libraries(ops fmt cublas utils)

target_include_directories(ops PUBLIC "include")
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
# "$<INSTALL_INTERFACE:${OPS_INSTALL_INCLUDE_DIR}>")

set_target_properties(ops
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS ops LIBRARY DESTINATION ${OPS_INSTALL_LIB_DIR})