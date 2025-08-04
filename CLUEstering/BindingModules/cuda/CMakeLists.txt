enable_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

# set the CUDA standard
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set_source_files_properties(binding_gpu_cuda.cpp PROPERTIES LANGUAGE CUDA)
pybind11_add_module(CLUE_GPU_CUDA SHARED binding_gpu_cuda.cpp)
target_include_directories(
  CLUE_GPU_CUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
                        ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(CLUE_GPU_CUDA PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED
                                                 CLUE_ENABLE_CACHING_ALLOCATOR)
target_compile_options(CLUE_GPU_CUDA PRIVATE --expt-relaxed-constexpr)
set_target_properties(
  CLUE_GPU_CUDA
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/
             CUDA_ARCHITECTURES "50;60;61;62;70;80;90")
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_GPU_CUDA
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_GPU_CUDA.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
