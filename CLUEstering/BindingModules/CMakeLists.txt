cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# include pybind11 extern subfolder
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_VERSION ">=3.8")

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  alpaka
  URL https://github.com/alpaka-group/alpaka/archive/refs/tags/1.2.0.tar.gz)
FetchContent_MakeAvailable(alpaka)

# create lib directory in CLUEstering folder
file(MAKE_DIRECTORY ../lib)

# Convolutional Kernels
pybind11_add_module(CLUE_Convolutional_Kernels SHARED
                    ${CMAKE_CURRENT_SOURCE_DIR}/binding_kernels.cpp)
target_include_directories(
  CLUE_Convolutional_Kernels
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(
  CLUE_Convolutional_Kernels PRIVATE ALPAKA_HOST_ONLY
                                     ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
set_target_properties(
  CLUE_Convolutional_Kernels
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../lib/CLUEstering/lib/)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_Convolutional_Kernels
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../lib/CLUEstering/lib/CLUE_Convolutional_Kernels.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/)

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
