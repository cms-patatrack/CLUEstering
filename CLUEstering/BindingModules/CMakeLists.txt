cmake_minimum_required(VERSION 3.16.0)

# Convolutional Kernels
pybind11_add_module(
  CLUE_Convolutional_Kernels SHARED
  ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_kernels.cpp)
target_include_directories(
  CLUE_Convolutional_Kernels
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${alpaka_SOURCE_DIR}/include
          ${Boost_INCLUDE_DIR})
target_compile_definitions(
  CLUE_Convolutional_Kernels PRIVATE ALPAKA_HOST_ONLY
                                     ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
set_target_properties(
  CLUE_Convolutional_Kernels
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_Convolutional_Kernels
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/CLUE_Convolutional_Kernels.*
    ${CMAKE_SOURCE_DIR}/CLUEstering/lib/
  COMMENT "Copying module to ${CMAKE_SOURCE_DIR}/CLUEstering/lib")


add_subdirectory(serial)

find_package(TBB)
if(TBB_FOUND)
  add_subdirectory(tbb)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_subdirectory(openmp)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

# GPU HIP
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_subdirectory(hip)
endif()

