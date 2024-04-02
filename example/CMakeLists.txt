set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
add_executable(example_test test.cc)
target_include_directories(example_test PRIVATE ${MET_CPU_KERNEL_INCLUDE} ${MET_GPU_KERNEL_INCLUDE} ${MET_UTILS_INCLUDE} ${MET_CORE_INCLUDE})

target_link_libraries(
  example_test
  core
  kernels
  utils
)

install(TARGETS example_test DESTINATION ${MET_INSTALL_BIN_DIR})
