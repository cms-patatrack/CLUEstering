cmake_minimum_required(VERSION 3.16.0)
project(CLUEstering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O0")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "-O2")
endif()

# include alpaka extern subfolder
include_directories(extern/alpaka/include)
# include pybind11 extern subfolder
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_VERSION ">=3.8")
add_subdirectory(extern/pybind11)

find_package(Boost 1.75.0)

# if boost is not found, it's fetched from the official boost repository
if(NOT Boost_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    boost
    URL https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
  )

  FetchContent_GetProperties(boost)
  if(NOT boost_POPULATED)
    FetchContent_Populate(boost)
  endif()
  set(Boost_PATH ./build/_deps/boost-src)
else()
  set(Boost_PATH ${Boost_INCLUDE_DIRS})
endif()

if(NOT DEFINED CPU_ONLY)
  set(CPU_ONLY OFF)
endif()
if(NOT DEFINED SERIAL_ONLY)
  set(SERIAL_ONLY OFF)
endif()

# create lib directory in CLUEstering folder
execute_process(COMMAND mkdir -p ./CLUEstering/lib)

# Convolutional Kernels compile convolutional kernel module
pybind11_add_module(CLUE_Convolutional_Kernels SHARED
                    ./CLUEstering/alpaka/BindingModules/binding_kernels.cc)
# link boost
target_link_libraries(CLUE_Convolutional_Kernels PRIVATE ${Boost_LIBRARIES})
target_include_directories(CLUE_Convolutional_Kernels PRIVATE ${Boost_PATH})
# alpaka build flags
target_compile_options(
  CLUE_Convolutional_Kernels
  PRIVATE -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND)
# set output directory
set_target_properties(
  CLUE_Convolutional_Kernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                        ./lib/CLUEstering/lib/)
# create link of shared object to lib folder inside CLUEstering directory
file(GLOB CONV_MODULE
     ./build/lib/CLUEstering/lib/CLUE_Convolutional_Kernels*.so)
foreach(CONV_MODULE ${CONV_MODULE})
  execute_process(COMMAND ln -sf ${CONV_MODULE}
                          ./CLUEstering/lib/CLUE_Convolutional_Kernels.so)
endforeach()

# CPU Serial compile cpu serial module
pybind11_add_module(CLUE_CPU_Serial SHARED
                    ./CLUEstering/alpaka/BindingModules/binding_cpu.cc)
# link boost
target_link_libraries(CLUE_CPU_Serial PRIVATE ${Boost_LIBRARIES})
target_include_directories(CLUE_CPU_Serial PRIVATE ${Boost_PATH})
# alpaka build flags
target_compile_options(
  CLUE_CPU_Serial
  PRIVATE -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND)
# set output directory
set_target_properties(CLUE_CPU_Serial PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                 ./lib/CLUEstering/lib/)
# create link of shared object to lib folder inside CLUEstering directory
file(GLOB CONV_MODULE ./build/lib/CLUEstering/lib/CLUE_CPU_Serial*.so)
foreach(CONV_MODULE ${CONV_MODULE})
  execute_process(COMMAND ln -sf ${CONV_MODULE}
                          ./CLUEstering/lib/CLUE_CPU_Serial.so)
endforeach()

if(NOT ${SERIAL_ONLY})
  find_package(TBB)

  # CPU TBB
  if(TBB_FOUND)
    # compile cpu tbb module
    pybind11_add_module(CLUE_CPU_TBB SHARED
                        ./CLUEstering/alpaka/BindingModules/binding_cpu_tbb.cc)
    target_link_libraries(CLUE_CPU_TBB PRIVATE ${Boost_LIBRARIES})
    target_include_directories(CLUE_CPU_TBB PRIVATE ${Boost_PATH})
    target_compile_options(
      CLUE_CPU_TBB
      PRIVATE -ltbb -DALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT
              -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
              -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND)
    target_link_libraries(CLUE_CPU_TBB PRIVATE TBB::tbb)
    set_target_properties(CLUE_CPU_TBB PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                  ./lib/CLUEstering/lib/)
    # create link of shared object to lib folder inside CLUEstering directory
    file(GLOB CONV_MODULE ./build/lib/CLUEstering/lib/CLUE_CPU_TBB*.so)
    foreach(CONV_MODULE ${CONV_MODULE})
      execute_process(COMMAND ln -sf ${CONV_MODULE}
                              ./CLUEstering/lib/CLUE_CPU_TBB.so)
    endforeach()
  endif()
endif()

if((NOT ${CPU_ONLY}) AND (NOT ${SERIAL_ONLY}))
  # check if CUDA is available
  include(CheckLanguage)
  check_language(CUDA)

  # GPU CUDA
  if(CMAKE_CUDA_COMPILER)
    # enable CUDA
    enable_language(CUDA)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

    # set the CUDA standard
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    # compile the file with .cc extension using nvcc
    set_source_files_properties(
      ./CLUEstering/alpaka/BindingModules/binding_gpu_cuda.cc
      PROPERTIES LANGUAGE CUDA)
    # compile gpu cuda module
    pybind11_add_module(CLUE_GPU_CUDA SHARED
                        ./CLUEstering/alpaka/BindingModules/binding_gpu_cuda.cc)
    # link boost
    target_link_libraries(CLUE_GPU_CUDA PRIVATE ${Boost_LIBRARIES})
    target_include_directories(CLUE_GPU_CUDA PRIVATE ${Boost_PATH})
    # set the cuda architectures
    set_target_properties(CLUE_GPU_CUDA PROPERTIES CUDA_ARCHITECTURES
                                                   "50;60;61;62;70")
    # alpaka build flags
    target_compile_options(
      CLUE_GPU_CUDA
      PRIVATE -DALPAKA_ACC_GPU_CUDA_PRESENT -DALPAKA_ACC_GPU_CUDA_ENABLED
              -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND)
    # nvcc compilation flags
    target_compile_options(
      CLUE_GPU_CUDA PRIVATE --expt-relaxed-constexpr -gencode
                            arch=compute_61,code=[sm_61,compute_61])
    # set output directory
    set_target_properties(CLUE_GPU_CUDA PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                   ./lib/CLUEstering/lib/)
    # create link of shared object to lib folder inside CLUEstering directory
    file(GLOB CONV_MODULE ./build/lib/CLUEstering/lib/CLUE_GPU_CUDA*.so)
    foreach(CONV_MODULE ${CONV_MODULE})
      execute_process(COMMAND ln -sf ${CONV_MODULE}
                              ./CLUEstering/lib/CLUE_GPU_CUDA.so)
    endforeach()
  endif()

  # GPU HIP check if HIP is available
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    # enable HIP
    enable_language(HIP)
    set(CMAKE_HIP_HOST_COMPILER ${CMAKE_HIP_COMPILER})

    # look for the hip package folder
    find_package(hip)

    set(hip_BASE "${hip_INCLUDE_DIRS}/..")
    # set the hipcc compiler
    set(CMAKE_CXX_COMPILER "${hip_BASE}/bin/hipcc")
    # compile gpu hip module
    pybind11_add_module(CLUE_GPU_HIP SHARED
                        ./CLUEstering/alpaka/BindingModules/binding_gpu_hip.cc)
    # link boost
    target_link_libraries(CLUE_GPU_HIP PRIVATE ${Boost_LIBRARIES})
    target_include_directories(CLUE_GPU_HIP PRIVATE ${Boost_PATH})
    # alpaka build flags
    target_compile_options(
      CLUE_GPU_HIP
      PRIVATE -DALPAKA_ACC_GPU_HIP_PRESENT -DALPAKA_ACC_GPU_HIP_ENABLED
              -DALPAKA_ACC_GPU_HIP_ASYNC_BACKEND)
    # link hip-rand libraries
    target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_INCLUDE_DIRS})
    target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_BASE}/hiprand/include)
    target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_BASE}/rocrand/include)
    # set output directory
    set_target_properties(CLUE_GPU_HIP PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                  ./lib/CLUEstering/lib/)
    # create link of shared object to lib folder inside CLUEstering directory
    file(GLOB CONV_MODULE ./build/lib/CLUEstering/lib/CLUE_GPU_HIP*.so)
    foreach(CONV_MODULE ${CONV_MODULE})
      execute_process(COMMAND ln -sf ${CONV_MODULE}
                              ./CLUEstering/lib/CLUE_GPU_HIP.so)
    endforeach()
  endif()
endif()
