file(GLOB BENCHMARK_SRC "*.cpp")

add_executable(benchmark ${BENCHMARK_SRC})
target_link_libraries(benchmark PRIVATE ops cublas utils)
# FIXME: bin/benchmark not able to run. may use rpath to solve ldd issue.
install(TARGETS benchmark DESTINATION ${MET_INSTALL_BIN_DIR})
