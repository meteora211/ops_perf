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
include(GoogleTest)


# TODO: expose BUILD_TEST in cmake configure
set(BUILD_TEST ON)

file(GLOB_RECURSE ALL_TEST_FILES *_test.cc)
if(BUILD_TEST)
  foreach(test_src ${ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    # set(test_name "${test_file_name}")
    message(STATUS ${test_file_name})
    add_executable(${test_file_name} "${test_src}")
    target_link_libraries(${test_file_name} core kernels utils gtest gtest_main)
    target_include_directories(${test_file_name} PRIVATE ${MET_CPU_KERNEL_INCLUDE} ${MET_GPU_KERNEL_INCLUDE} ${MET_UTILS_INCLUDE})
    add_test(NAME ${test_file_name} COMMAND $<TARGET_FILE:${test_file_name}>)
    # gtest_discover_tests(${test_file_name})
    # if(INSTALL_TEST)
    #   install(TARGETS ${test_file_name} DESTINATION test)
    # endif()
  endforeach()
endif()
