add_subdirectory(utils)
# add_subdirectory(benchmark)

set(MET_CPU_KERNEL_INCLUDE)
set(MET_GPU_KERNEL_INCLUDE)
set(MET_UTILS_INCLUDE)
set(MET_CORE_INCLUDE)

list(APPEND MET_CPU_KERNEL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cpu)
list(APPEND MET_GPU_KERNEL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
list(APPEND MET_UTILS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
list(APPEND MET_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/core)

file(GLOB MET_CORE_HEADERS "core/*.h")
file(GLOB MET_CORE_SRCS "core/*.cc")

add_library(core SHARED ${MET_CORE_SRCS})
target_include_directories(core PUBLIC ${MET_CORE_INCLUDE} ${MET_CPU_KERNEL_INCLUDE} ${MET_UTILS_INCLUDE})

install(TARGETS core LIBRARY DESTINATION ${MET_INSTALL_LIB_DIR})

file(GLOB CPU_KERNELS "cpu/*.cc")
file(GLOB CUDA_KERNELS "gpu/*.cu")


add_library(kernels SHARED ${CPU_KERNELS} ${CUDA_KERNELS})
target_link_libraries(kernels PRIVATE fmt cudnn cublas utils core)

target_include_directories(kernels PUBLIC ${MET_UTILS_INCLUDE} ${MET_CORE_INCLUDE})
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
# "$<INSTALL_INTERFACE:${MET_INSTALL_INCLUDE_DIR}>")

set_target_properties(kernels
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS kernels LIBRARY DESTINATION ${MET_INSTALL_LIB_DIR})

set(MET_CPU_KERNEL_INCLUDE ${MET_CPU_KERNEL_INCLUDE} PARENT_SCOPE)
set(MET_GPU_KERNEL_INCLUDE ${MET_GPU_KERNEL_INCLUDE} PARENT_SCOPE)
set(MET_UTILS_INCLUDE ${MET_UTILS_INCLUDE} PARENT_SCOPE)
set(MET_CORE_INCLUDE ${MET_CORE_INCLUDE} PARENT_SCOPE)
