enable_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set_source_files_properties(../test_clustering.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(../test_host_points.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(../test_device_points.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cuda.out ../test_clustering.cpp ../test_host_points.cpp ../test_device_points.cpp)
target_include_directories(
  cuda.out
  PRIVATE ${CMAKE_SURCE_DIR}../../include/CLUEstering
          ${doctest_SOURCE_DIR}/doctest ${alpaka_SOURCE_DIR}/include
          ${Boost_INCLUDE_DIR})
target_compile_definitions(cuda.out PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
target_compile_options(cuda.out PRIVATE --expt-relaxed-constexpr)
set_target_properties(cuda.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                          CUDA_ARCHITECTURES "50;60;61;62;70")
