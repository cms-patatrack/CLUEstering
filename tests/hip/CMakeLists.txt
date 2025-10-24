enable_language(HIP)
find_package(HIP)

file(GLOB sources ../*.cpp)
foreach(src IN LISTS sources)
  set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
endforeach()

foreach(source ${sources})
  get_filename_component(test_name ${source} NAME_WE)

  add_executable(${test_name} ${sources})
  target_include_directories(
    ${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/../include
                         ${doctest_SOURCE_DIR}/doctest ${HIP_INCLUDE_DIR})
  target_link_libraries(${test_name} PRIVATE alpaka::alpaka Boost::boost
                                             fmt::fmt)
  target_compile_definitions(${test_name} PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED
                                                  CLUE_ENABLE_CACHING_ALLOCATOR)
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/hip)
endforeach()
