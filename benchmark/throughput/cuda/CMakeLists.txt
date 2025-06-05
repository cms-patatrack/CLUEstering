
enable_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set_source_files_properties(${CMAKE_SOURCE_DIR}/worker.cpp PROPERTIES LANGUAGE CUDA)
add_library(cuda SHARED ${CMAKE_SOURCE_DIR}/worker.cpp)
target_link_libraries(cuda PRIVATE alpaka::alpaka Boost::boost)
target_include_directories(
    cuda PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                       ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(cuda PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
target_compile_options(cuda PRIVATE --expt-relaxed-constexpr)
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON
										  CUDA_ARCHITECTURES "50;60;61;62;70;80;90")

