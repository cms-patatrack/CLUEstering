cmake_minimum_required(VERSION 3.16.0)
project(CLUEstering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O2")

# include alpaka extern subfolder
include_directories(extern/alpaka/include)
# include pybind11 extern subfolder
add_subdirectory(extern/pybind11)

find_package(Boost 1.75.0)

if (NOT Boost_FOUND)
  include(FetchContent)
  FetchContent_Declare(
	boost
	URL https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
  )

  FetchContent_GetProperties(boost)
  if (NOT boost_POPULATED)
    FetchContent_Populate(boost)
  endif()
  set(Boost_PATH ./build/_deps/boost-src)
else()
  set(Boost_PATH ${Boost_INCLUDE_DIRS})
endif()

# compile convolutional kernel module
pybind11_add_module(CLUE_Convolutional_Kernels SHARED
                    ./CLUEstering/alpaka/BindingModules/binding_kernels.cc)
target_link_libraries(CLUE_Convolutional_Kernels PRIVATE ${Boost_LIBRARIES})
target_include_directories(CLUE_Convolutional_Kernels PRIVATE ${Boost_PATH})
target_compile_options(
  CLUE_Convolutional_Kernels
  PRIVATE -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND)
set_target_properties(CLUE_Convolutional_Kernels
                      PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../CLUEstering/)

# compile cpu serial module
pybind11_add_module(CLUE_CPU_Serial SHARED
                    ./CLUEstering/alpaka/BindingModules/binding_cpu.cc)
				  target_link_libraries(CLUE_CPU_Serial PRIVATE ${Boost_LIBRARIES})
target_include_directories(CLUE_CPU_Serial PRIVATE ${Boost_PATH})
target_compile_options(
  CLUE_CPU_Serial
  PRIVATE -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND)
set_target_properties(CLUE_CPU_Serial PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                 ../CLUEstering/)

find_package(TBB)

if (TBB_FOUND)
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
												../CLUEstering/)
endif()

include(CheckLanguage)
# check if CUDA is available
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  # enable CUDA
  enable_language(CUDA)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

  # set the CUDA standard
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  set(CMAKE_CUDA_FLAGS "-Wall -Wextra -g -O2")

  # compile nvidia gpu async module
  pybind11_add_module(CLUE_GPU_CUDA SHARED
					  ./CLUEstering/alpaka/BindingModules/binding_gpu_cuda.cu)
  target_link_libraries(CLUE_Convolutional_Kernels PRIVATE ${Boost_LIBRARIES})
  target_include_directories(CLUE_Convolutional_Kernels PRIVATE ${Boost_PATH})
  # set the cuda architectures
  set_target_properties(CLUE_GPU_CUDA PROPERTIES CUDA_ARCHITECTURES
												 "50;60;61;62;70")
  # alpaka compilation flags
  target_compile_options(
	CLUE_GPU_CUDA
	PRIVATE -DALPAKA_ACC_GPU_CUDA_PRESENT -DALPAKA_ACC_GPU_CUDA_ENABLED
			-DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND)
  # nvcc compilation flags
  target_compile_options(
	CLUE_GPU_CUDA PRIVATE --expt-relaxed-constexpr -gencode
                        arch=compute_61,code=[sm_61,compute_61])
  set_target_properties(CLUE_GPU_CUDA PROPERTIES LIBRARY_OUTPUT_DIRECTORY
												 ../CLUEstering/)
endif()
