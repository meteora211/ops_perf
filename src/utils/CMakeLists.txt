file(GLOB UTILS "*.cpp")
add_library(utils SHARED ${UTILS})
target_link_libraries(utils fmt)
# target_include_directories(utils PUBLIC "include")
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
# "$<INSTALL_INTERFACE:${OPS_INSTALL_INCLUDE_DIR}>")

install(TARGETS utils LIBRARY DESTINATION ${OPS_INSTALL_LIB_DIR})
