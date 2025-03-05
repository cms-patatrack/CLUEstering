cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(APPEND CMAKE_CXX_FLAGS_DEBUG
       " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -Wall -Wextra -pg")
string(APPEND CMAKE_CXX_FLAGS "-O2 -funroll-loops -ffast-math -funsafe-math-optimizations -fopt-info-vec -ftree-vectorize -march=native")

find_package(Boost 1.75.0 REQUIRED)

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
