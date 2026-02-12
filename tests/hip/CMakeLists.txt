enable_language(HIP)
find_package(HIP)

file(GLOB sources ${CMAKE_SOURCE_DIR}/tests/*.cpp)
foreach(src IN LISTS sources)
  set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
endforeach()

foreach(source ${sources})
  get_filename_component(test_name ${source} NAME_WE)
  set(target hip_${test_name})

  add_executable(${target} ${source})
  add_test(NAME ${target} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hip/${test_name})
  target_include_directories(
    ${target} PRIVATE ${CMAKE_SOURCE_DIR}/include ${doctest_SOURCE_DIR}
                      ${HIP_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE alpaka::alpaka Boost::atomic fmt::fmt)
  target_compile_definitions(
    ${target} PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED CLUE_ENABLE_CACHING_ALLOCATOR
                      TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hip
               OUTPUT_NAME ${test_name})
endforeach()
