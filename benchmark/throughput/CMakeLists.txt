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
find_package(TBB REQUIRED)
find_package(pybind11)

if(pybind11_FOUND)
  set(pybind_link pybind11::embed)
  add_definitions(-DPYBIND11)
else()
  set(pybind_link "")
  message(STATUS "pybind11 not found, so no plots will be generated.")
endif()

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
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

set(benchmark_include_dirs ${CMAKE_SOURCE_DIR}/../../include
                           ${CMAKE_SOURCE_DIR}/..)

set(benchmark_defs ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ALPAKA_HOST_ONLY)
set(benchmark_libs serial alpaka::alpaka Boost::boost TBB::tbb)

if(CMAKE_CUDA_COMPILER)
  list(APPEND benchmark_defs CLUE_HAS_CUDA)
  list(APPEND benchmark_libs cuda)
endif()

add_executable(throughput.out main.cpp)
target_include_directories(throughput.out PRIVATE ${benchmark_include_dirs})
target_link_libraries(throughput.out PRIVATE ${benchmark_libs})
target_compile_definitions(throughput.out PRIVATE ${benchmark_defs})
