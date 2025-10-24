enable_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB sources ../*.cpp)

foreach(src IN LISTS sources)
  set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
endforeach()

foreach(source ${sources})
  get_filename_component(test_name ${source} NAME_WE)

  add_executable(${test_name} ${source})
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                                  ${doctest_SOURCE_DIR}/doctest)
  target_link_libraries(${test_name} PRIVATE alpaka::alpaka Boost::boost
                                             fmt::fmt)
  target_compile_definitions(${test_name} PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED
                                                  CLUE_ENABLE_CACHING_ALLOCATOR)
  target_compile_options(${test_name} PRIVATE --expt-relaxed-constexpr)
  set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                CUDA_ARCHITECTURES "75;80;90")
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/cuda)
endforeach()
