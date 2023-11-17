cmake_minimum_required(VERSION 3.2)

project(matmul)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -march=native -fopenmp")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(OPS_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(OPS_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(OPS_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OPS_INSTALL_INCLUDE_DIR})
include_directories(${OPS_HEADERS_DIR})
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
add_subdirectory(temp_source)
# add_subdirectory(tests)
