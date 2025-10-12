enable_language(HIP)
find_package(HIP)

file(GLOB sources ../*.cpp)
foreach(src IN LISTS sources)
  set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
endforeach()

add_executable(hip.out ${sources})
target_include_directories(
  hip.out PRIVATE ${CMAKE_SOURCE_DIR}/../include ${doctest_SOURCE_DIR}/doctest
                  ${HIP_INCLUDE_DIR})
target_link_libraries(hip.out PRIVATE alpaka::alpaka Boost::boost fmt::fmt)
target_compile_definitions(hip.out PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED
                                           CLUE_ENABLE_CACHING_ALLOCATOR)
