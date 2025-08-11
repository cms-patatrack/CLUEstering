enable_language(HIP)
find_package(HIP)

file(GLOB sources ../*.cpp)
foreach(src IN LISTS sources)
  set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
endforeach()

add_executable(hip.out ${sources})
target_link_libraries(hip.out PRIVATE fmt::fmt)
target_include_directories(
  hip.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../include ${doctest_SOURCE_DIR}/doctest
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR} ${HIP_INCLUDE_DIR})
target_compile_definitions(hip.out PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED
                                           CLUE_ENABLE_CACHING_ALLOCATOR)
