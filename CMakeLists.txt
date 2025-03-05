cmake_minimum_required(VERSION 3.16.0)
project(CLUEstering LANGUAGES CXX)

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
string(APPEND CMAKE_CXX_FLAGS "-O2 -funroll-loops -ffast-math -funsafe-math-optimizations -ftree-vectorize -march=native")

# include pybind11 extern subfolder
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_VERSION ">=3.8")
add_subdirectory(extern/pybind11)

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  alpaka
  URL https://github.com/alpaka-group/alpaka/archive/refs/tags/1.2.0.tar.gz)
FetchContent_MakeAvailable(alpaka)

# create lib directory in CLUEstering folder
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/CLUEstering/lib)

# Convolutional Kernels
pybind11_add_module(
  CLUE_Convolutional_Kernels SHARED
  ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_kernels.cpp)
target_include_directories(
  CLUE_Convolutional_Kernels
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${alpaka_SOURCE_DIR}/include
          ${Boost_INCLUDE_DIR})
target_compile_definitions(
  CLUE_Convolutional_Kernels PRIVATE ALPAKA_HOST_ONLY
                                     ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
set_target_properties(
  CLUE_Convolutional_Kernels
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_Convolutional_Kernels
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/CLUE_Convolutional_Kernels.*
    ${CMAKE_SOURCE_DIR}/CLUEstering/lib/
  COMMENT "Copying module to ${CMAKE_SOURCE_DIR}/CLUEstering/lib")

# CPU Serial
pybind11_add_module(
  CLUE_CPU_Serial SHARED
  ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_cpu.cpp)
target_include_directories(
  CLUE_CPU_Serial PRIVATE ${CMAKE_SOURCE_DIR}/include
                          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(
  CLUE_CPU_Serial PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
set_target_properties(
  CLUE_CPU_Serial PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                             ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_Serial
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/CLUE_CPU_Serial.*
    ${CMAKE_SOURCE_DIR}/CLUEstering/lib/
  COMMENT "Copying module to ${CMAKE_SOURCE_DIR}/CLUEstering/lib")

find_package(TBB)
# CPU TBB
if(TBB_FOUND)
  pybind11_add_module(
    CLUE_CPU_TBB SHARED
    ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_cpu_tbb.cpp)
  target_include_directories(
    CLUE_CPU_TBB PRIVATE ${CMAKE_SOURCE_DIR}/include
                         ${alpaka_SOURCE_DIR}/include ${Boost_SOURCE_DIR})
  target_compile_definitions(
    CLUE_CPU_TBB PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  target_link_libraries(CLUE_CPU_TBB PRIVATE TBB::tbb)
  set_target_properties(
    CLUE_CPU_TBB PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                            ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/)
  # copy shared library for local testing
  add_custom_command(
    TARGET CLUE_CPU_TBB
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/CLUE_CPU_TBB.*
      ${CMAKE_SOURCE_DIR}/CLUEstering/lib/
    COMMENT "Copying module to ${CMAKE_SOURCE_DIR}/CLUEstering/lib")
endif()

find_package(OpenMP)
# CPU OpenMP
if(OpenMP_CXX_FOUND)
  pybind11_add_module(
    CLUE_CPU_OMP SHARED
    ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_cpu_omp.cpp)
  target_include_directories(
    CLUE_CPU_OMP PRIVATE ${CMAKE_SOURCE_DIR}/include
                         ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
  target_compile_definitions(
    CLUE_CPU_OMP PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
  target_link_libraries(CLUE_CPU_OMP PRIVATE OpenMP::OpenMP_CXX)
  set_target_properties(
    CLUE_CPU_OMP PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                            ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/)
  # copy shared library for local testing
  add_custom_command(
    TARGET CLUE_CPU_OMP
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/CLUE_CPU_OMP.*
      ${CMAKE_SOURCE_DIR}/CLUEstering/lib/
    COMMENT "Copying module to ${CMAKE_SOURCE_DIR}/CLUEstering/lib")
endif()

# check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

# GPU CUDA
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

  # set the CUDA standard
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_gpu_cuda.cc
    PROPERTIES LANGUAGE CUDA)
  pybind11_add_module(
    CLUE_GPU_CUDA SHARED
    ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_gpu_cuda.cpp)
  target_include_directories(
    CLUE_GPU_CUDA PRIVATE ${CMAKE_SOURCE_DIR}/include
                          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
  target_compile_definitions(CLUE_GPU_CUDA PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
  target_compile_options(CLUE_GPU_CUDA PRIVATE --expt-relaxed-constexpr)
  set_target_properties(
    CLUE_GPU_CUDA
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/
               CUDA_ARCHITECTURES "50;60;61;62;70")
  # copy shared library for local testing
  add_custom_command(
    TARGET CLUE_GPU_CUDA
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/CLUE_GPU_CUDA.*
      ${CMAKE_SOURCE_DIR}/CLUEstering/lib/
    COMMENT "Copying module to ${CMAKE_SOURCE_DIR}/CLUEstering/lib")
endif()

# GPU HIP
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  set(CMAKE_HIP_HOST_COMPILER ${CMAKE_HIP_COMPILER})

  find_package(hip)

  set(hip_BASE "${hip_INCLUDE_DIRS}/..")
  set(CMAKE_CXX_COMPILER "${hip_BASE}/bin/hipcc")
  pybind11_add_module(
    CLUE_GPU_HIP SHARED
    ${CMAKE_SOURCE_DIR}/CLUEstering/BindingModules/binding_gpu_hip.cpp)
  target_include_directories(
    CLUE_GPU_HIP PRIVATE ${CMAKE_SOURCE_DIR}/include
                         ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
  target_compile_options(CLUE_GPU_HIP PRIVATE -DALPAKA_ACC_GPU_HIP_ENABLED)
  target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_INCLUDE_DIRS})
  target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_BASE}/hiprand/include)
  target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_BASE}/rocrand/include)
  set_target_properties(
    CLUE_GPU_HIP PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                            ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/)
  # create link of shared object to lib folder inside CLUEstering directory
  add_custom_command(
    TARGET CLUE_GPU_HIP
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/lib/CLUEstering/lib/CLUE_GPU_HIP.*
      ${CMAKE_SOURCE_DIR}/CLUEstering/lib/
    COMMENT "Copying module to ${CMAKE_SOURCE_DIR}/CLUEstering/lib")
endif()
