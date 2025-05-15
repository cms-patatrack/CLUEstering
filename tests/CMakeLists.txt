cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(APPEND CMAKE_CXX_FLAGS_DEBUG
       " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -Wall -Wextra -pg -fsanitize=address")
string(APPEND CMAKE_CXX_FLAGS "-O2")

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)

# alpaka
FetchContent_Declare(
  alpaka
  GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
  GIT_TAG develop
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(alpaka)

# doctest
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG dev)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_MakeAvailable(doctest)
endif()

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.2.1  # or another stable version
)
FetchContent_MakeAvailable(fmt)

add_subdirectory(cpu)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()
