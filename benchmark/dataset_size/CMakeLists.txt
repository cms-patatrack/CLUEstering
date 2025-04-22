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
  " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -Wall -Wextra -Wpedantic -Wpedantic -Werror"
)
string(
  APPEND
  CMAKE_CXX_FLAGS
  "-O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

find_package(Boost 1.75.0 REQUIRED)
find_package(pybind11)

if(pybind11_FOUND)
  set(pybind_link pybind11::embed)
  add_definitions(-DPYBIND11)
else()
  set(pybind_link "")
  message(STATUS "pybind11 not found, so no plots will be generated.")
endif()

include(FetchContent)
FetchContent_Declare(
  alpaka
  URL https://github.com/alpaka-group/alpaka/archive/refs/tags/1.2.0.tar.gz)

FetchContent_MakeAvailable(alpaka)

add_subdirectory(cpu)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
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
