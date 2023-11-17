function(CompileCUDA SOURCE_FILES)
  get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  foreach(source_file ${SOURCE_FILES})
    get_filename_component(executable_name ${source_file} NAME_WE)
    add_executable(${executable_name} ${source_file})
    set_target_properties(${executable_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    # target_compile_options(${executable_name} PRIVATE -O3 -Wall -fno-elide-constructors)
    install(TARGETS ${executable_name} DESTINATION ${OPS_INSTALL_BIN_DIR})
  endforeach()
endfunction()