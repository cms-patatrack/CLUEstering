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
  set(target cuda_${test_name})

  add_executable(${target} ${source})
  add_test(NAME ${target} COMMAND ${CMAKE_BINARY_DIR}/cuda/${test_name})
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                               ${doctest_SOURCE_DIR}/doctest)
  target_link_libraries(${target} PRIVATE alpaka::alpaka Boost::atomic fmt::fmt)
  target_compile_definitions(${target} PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED
                                               CLUE_ENABLE_CACHING_ALLOCATOR)
  target_compile_options(${target} PRIVATE --expt-relaxed-constexpr)
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                             CUDA_ARCHITECTURES "75;80;90")
  set_target_properties(
    ${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda
                         OUTPUT_NAME ${test_name})
endforeach()
