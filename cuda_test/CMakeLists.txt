file(GLOB SOURCE_FILES "*.cu")
CompileCUDA("${SOURCE_FILES}")
add_subdirectory(micro_benchmark)
