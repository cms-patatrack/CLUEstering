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
add_executable(cuda.out ${sources})
target_link_libraries(cuda.out PRIVATE fmt::fmt)
target_include_directories(
  cuda.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../include
          ${doctest_SOURCE_DIR}/doctest
          ${alpaka_SOURCE_DIR}/include
          ${Boost_INCLUDE_DIR})
target_compile_definitions(cuda.out PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED
                                            CLUE_ENABLE_CACHING_ALLOCATOR)
target_compile_options(cuda.out PRIVATE --expt-relaxed-constexpr)
set_target_properties(
  cuda.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES
                                                    "50;60;61;62;70;80;90")
