include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG release-1.12.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  # URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  baseline_test
  baseline_test.cc
)


target_include_directories(baseline_test PRIVATE ${CPU_OPS_INCLUDE} ${GPU_OPS_INCLUDE} ${OPS_UTILS_INCLUDE})

target_link_libraries(
  baseline_test
  ops
  utils
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(baseline_test)
