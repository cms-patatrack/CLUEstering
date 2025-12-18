cmake_minimum_required(VERSION 3.10)

project(NativeHIPCLUEsteringExample LANGUAGES CXX HIP)

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

include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_executable(native_hip.out main.cpp)
  target_link_libraries(native_hip.out PRIVATE CLUEstering::CLUEstering
                                               alpaka::alpaka)
  target_compile_definitions(native_hip.out PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)
else()
  message(FATAL_ERROR "CUDA not found. Please install it.")
endif()
