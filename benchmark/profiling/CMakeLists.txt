cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(
  APPEND
  CMAKE_CXX_FLAGS_DEBUG
  " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -O0 -Wall -Wextra -Wpedantic -Wshadow -Wimplicit-fallthrough -Wextra-semi -Wold-style-cast -g -pg -fsanitize=address"
)
set(CMAKE_CXX_FLAGS_RELEASE
    " -O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

find_package(Boost 1.75.0 REQUIRED COMPONENTS atomic)

find_package(alpaka)
if(NOT alpaka_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    alpaka
    URL https://github.com/alpaka-group/alpaka/archive/refs/tags/1.2.0.tar.gz)
  FetchContent_MakeAvailable(alpaka)
endif()

add_subdirectory(cpu)

include(CheckLanguage)

execute_process(
  COMMAND nvidia-smi --query-gpu=name --format=csv,noheader
  OUTPUT_VARIABLE GPU_LIST
  ERROR_QUIET
  RESULT_VARIABLE SMI_STATUS)

if(SMI_STATUS EQUAL 0 AND NOT GPU_LIST STREQUAL "")
  set(NVIDIA_GPU_PRESENT TRUE)
else()
  set(NVIDIA_GPU_PRESENT FALSE)
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NVIDIA_GPU_PRESENT)
  add_subdirectory(cuda)
endif()

execute_process(
  COMMAND rocm-smi --showproductname
  OUTPUT_VARIABLE AMD_GPU_LIST
  ERROR_QUIET
  RESULT_VARIABLE SMI_STATUS)

if(SMI_STATUS EQUAL 0 AND NOT AMD_GPU_LIST STREQUAL "")
  set(AMD_GPU_PRESENT TRUE)
else()
  set(AMD_GPU_PRESENT FALSE)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER AND AMD_GPU_PRESENT)
  add_subdirectory(hip)
endif()

set(_sycl_search_dirs ${SYCL_ROOT_DIR} /usr/lib /usr/local/lib
                      /opt/intel/oneapi/compiler/latest/linux)
find_program(
  SYCL_COMPILER
  NAMES icpx
  HINTS ${_sycl_search_dirs}
  PATH_SUFFIXES bin)
find_path(
  SYCL_INCLUDE_DIR
  NAMES sycl/sycl.hpp
  HINTS ${_sycl_search_dirs}
  PATH_SUFFIXES include)
find_path(
  SYCL_LIB_DIR
  NAMES libsycl.so
  HINTS ${_sycl_search_dirs}
  PATH_SUFFIXES lib)
find_package(oneDPL)

if(oneDPL_FOUND
   AND SYCL_COMPILER
   AND SYCL_INCLUDE_DIR
   AND SYCL_LIB_DIR)
  add_subdirectory(sycl)
endif()
