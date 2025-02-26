cmake_minimum_required(VERSION 3.16.0)
project(TestPoints LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(APPEND CMAKE_CXX_FLAGS_DEBUG
       " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -Wall -Wextra -pg")
string(APPEND CMAKE_CXX_FLAGS "-O2")

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  alpaka
  GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
  GIT_TAG develop
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(alpaka)

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
target_include_directories(
  serial.out
  PRIVATE ${CMAKE_SURCE_DIR}../../../CLUEstering ${doctest_SOURCE_DIR}/doctest
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(
  serial.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

find_package(TBB)
if(TBB_FOUND)
  add_executable(tbb.out TestTilesExternal.cpp)
  target_include_directories(
    tbb.out
    PRIVATE ${CMAKE_SURCE_DIR}../../../CLUEstering
            ${doctest_SOURCE_DIR}/doctest ${alpaka_SOURCE_DIR}/include
            ${Boost_INCLUDE_DIR})
  target_link_libraries(tbb.out PRIVATE TBB::tbb)
  target_compile_definitions(tbb.out PRIVATE ALPAKA_HOST_ONLY
                                             ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
endif()

find_package(OpenMP)
# CPU OpenMP
if(OpenMP_CXX_FOUND)
  add_executable(openmp.out TestTilesExternal.cpp)
  target_include_directories(
    openmp.out
    PRIVATE ${CMAKE_SURCE_DIR}../../../CLUEstering
            ${doctest_SOURCE_DIR}/doctest ${alpaka_SOURCE_DIR}/include
            ${Boost_INCLUDE_DIR})
  target_link_libraries(openmp.out PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(
    openmp.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
endif()

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  add_executable(cuda.out TestTilesExternal.cpp)
  target_include_directories(
    cuda.out
    PRIVATE ${CMAKE_SURCE_DIR}../../../CLUEstering
            ${doctest_SOURCE_DIR}/doctest ${alpaka_SOURCE_DIR}/include
            ${Boost_INCLUDE_DIR})
  target_compile_options(cuda.out PRIVATE -DALPAKA_HOST_ONLY
                                          -DALPAKA_ACC_GPU_CUDA_ENABLED)
endif()
