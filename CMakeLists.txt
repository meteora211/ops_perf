cmake_minimum_required(VERSION 3.2)

project(ops_perf)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
# TODO: debug option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O0 -g -march=native -fopenmp")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(MET_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(MET_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(MET_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

include_directories(${MET_INSTALL_INCLUDE_DIR})
include_directories(${MET_HEADERS_DIR})
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompileCUDA)

include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 10.0.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(fmt)

add_subdirectory(src)
# add_subdirectory(temp_source)
add_subdirectory(tests)
add_subdirectory(example)
