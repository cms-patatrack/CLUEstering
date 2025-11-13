cmake_minimum_required(VERSION 3.10)

project(CLUEsteringExample)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CLUEstering REQUIRED)
find_package(alpaka REQUIRED)
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
  target_link_libraries(serial.out PRIVATE CLUEstering::CLUEstering
                                           alpaka::alpaka)
endif()
if(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  find_package(TBB REQUIRED)

  if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB not found. Please install it.")
  endif()
  add_executable(tbb.out main.cpp)
  target_compile_definitions(tbb.out PRIVATE ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  target_link_libraries(tbb.out PRIVATE CLUEstering::CLUEstering alpaka::alpaka
                                        TBB::tbb)
endif()
if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
  find_package(OpenMP REQUIRED)

  if(NOT TARGET OpenMP::OpenMP_CXX)
    message(FATAL_ERROR "OpenMP not found. Please install it.")
  endif()
  add_executable(openmp.out main.cpp)
  target_compile_definitions(openmp.out
                             PRIVATE ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
  target_link_libraries(openmp.out PRIVATE CLUEstering::CLUEstering
                                           alpaka::alpaka OpenMP::OpenMP_CXX)
endif()
if(ALPAKA_ACC_GPU_CUDA_ENABLED)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(cuda.out main.cpp)
    target_link_libraries(cuda.out PRIVATE CLUEstering::CLUEstering
                                           alpaka::alpaka)
    target_compile_definitions(cuda.out PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
    target_compile_options(cuda.out PRIVATE --expt-relaxed-constexpr)
    set_target_properties(cuda.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                              CUDA_ARCHITECTURES "75;80;90")
  else()
    message(FATAL_ERROR "CUDA not found. Please install it.")
  endif()
endif()
