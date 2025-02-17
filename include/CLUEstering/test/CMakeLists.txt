cmake_minimum_required(VERSION 3.16.0)
project(TestPoints LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O0")
endif()

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  alpaka
  GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
  GIT_TAG develop
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(alpaka)

include_directories(../../../CLUEstering)

include(FetchContent)
# Get doctest
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_MakeAvailable(doctest)
endif()

add_executable(serial.out TestTilesExternal.cpp)
target_include_directories(serial.out SYSTEM
                           PRIVATE ${doctest_SOURCE_DIR}/doctest)
target_include_directories(serial.out PRIVATE ${alpaka_SOURCE_DIR}/include
                                              ${Boost_INCLUDE_DIR})
target_compile_options(serial.out PRIVATE -DALPAKA_HOST_ONLY
                                          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

add_executable(tbb.out TestTilesExternal.cpp)
target_include_directories(tbb.out SYSTEM PRIVATE ${doctest_SOURCE_DIR}/doctest)
target_include_directories(tbb.out PRIVATE ${alpaka_SOURCE_DIR}/include
                                           ${Boost_INCLUDE_DIR})
target_compile_options(tbb.out PRIVATE -DALPAKA_HOST_ONLY
                                       -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
add_executable(cuda.out TestTilesExternal.cpp)
target_include_directories(cuda.out SYSTEM PRIVATE ${doctest_SOURCE_DIR}/doctest)
target_include_directories(cuda.out PRIVATE ${alpaka_SOURCE_DIR}/include
                                           ${Boost_INCLUDE_DIR})
target_compile_options(cuda.out PRIVATE -DALPAKA_HOST_ONLY
										-DALPAKA_ACC_GPU_CUDA_ENABLED)
endif()
