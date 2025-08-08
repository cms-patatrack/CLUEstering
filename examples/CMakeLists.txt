cmake_minimum_required(VERSION 3.10)
project(CLUEsteringExample)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CLUEstering)
if(NOT CLUEstering_FOUND)
  message(
    FATAL_ERROR
      "CLUEstering not found. This example requires it to be installed. Please install following the instructions in the README."
  )
endif()

if(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
  add_executable(serial.out main.cpp)
  target_compile_definitions(serial.out
                             PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
endif()
if(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  add_executable(tbb.out main.cpp)
  target_compile_definitions(tbb.out PRIVATE ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
endif()
if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
  add_executable(openmp.out main.cpp)
  target_compile_definitions(openmp.out
                             PRIVATE ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
endif()
if(ALPAKA_ACC_GPU_CUDA_ENABLED)
  add_executable(cuda.out main.cpp)
  target_compile_definitions(cuda.out PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
endif()
