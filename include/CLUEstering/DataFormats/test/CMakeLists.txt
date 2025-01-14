cmake_minimum_required(VERSION 3.16.0)
project(TestPoints LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wpedantic -g -O0")
endif()

include_directories(..)

include(FetchContent)
# Get doctest
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_MakeAvailable(doctest)
endif()

# CPU Serial
add_executable(test_points.out TestPoints.cpp)
target_include_directories(test_points.out SYSTEM PRIVATE ${doctest_SOURCE_DIR}/doctest)
