cmake_minimum_required(VERSION 3.16.0)
project(Profiling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O0 -pg")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "-O2")
endif()

include(FetchContent)

find_package(alpaka)
if (NOT alpaka_FOUND)
	FetchContent_Declare(
		alpaka
		GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
		GIT_TAG develop
	)

	FetchContent_GetProperties(alpaka)
	if(NOT alpaka_POPULATED)
		FetchContent_Populate(alpaka)
	endif()
	set(alpaka_PATH ./build/_deps/alpaka-src/include)
else()
	set(alpaka_PATH ${alpaka_INCLUDE_DIRS})
endif()


# look for boost
find_package(Boost 1.75.0)
# if boost is not found, it's fetched from the official boost repository
if(NOT Boost_FOUND)
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

# include source directories
include_directories(../../inlcude/)

# CPU Serial
add_executable(serial.out main.cpp)
target_include_directories(serial.out PRIVATE ${alpaka_PATH})
# link boost
target_link_libraries(serial.out PRIVATE ${Boost_LIBRARIES})
target_include_directories(serial.out PRIVATE ${Boost_PATH})
# alpaka build flags
target_compile_options(
  serial.out
  PRIVATE -DALPAKA_HOST_ONLY -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND)
if(NOT "${ANNOTATE}" STREQUAL "")
  target_compile_options(serial.out PRIVATE -DANNOTATE)
endif()
target_compile_options(serial.out PRIVATE -pg)
target_link_options(serial.out PRIVATE -pg)

if(NOT ${SERIAL_ONLY})
  find_package(TBB)

  # CPU TBB
  if(TBB_FOUND)
    # compile cpu tbb module
    add_executable(tbb.out main.cpp)
	target_include_directories(tbb.out PRIVATE ${alpaka_PATH})
    target_link_libraries(tbb.out PRIVATE ${Boost_LIBRARIES})
    target_include_directories(tbb.out PRIVATE ${Boost_PATH})
    target_compile_options(
      tbb.out
      PRIVATE -ltbb -DALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT
              -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
              -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND)
    target_link_libraries(tbb.out PRIVATE TBB::tbb)
    if(NOT "${ANNOTATE}" STREQUAL "")
      target_compile_options(tbb.out PRIVATE -DANNOTATE)
    endif()
    target_compile_options(serial.out PRIVATE -pg)
    target_link_options(serial.out PRIVATE -pg)
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

    set(CMAKE_CUDA_FLAGS "-Wall -Wextra -g -O2")

    # compile the file with .cc extension using nvcc
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    # compile gpu cuda module
    add_executable(cuda.out main.cpp)
	target_include_directories(cuda.out PRIVATE ${alpaka_PATH})
    # link boost
    target_link_libraries(cuda.out PRIVATE ${Boost_LIBRARIES})
    target_include_directories(cuda.out PRIVATE ${Boost_PATH})
    # set the cuda architectures
    set_target_properties(cuda.out PROPERTIES CUDA_ARCHITECTURES
                                              "50;60;61;62;70")
    # alpaka build flags
    target_compile_options(
      cuda.out
      PRIVATE -DALPAKA_ACC_GPU_CUDA_PRESENT -DALPAKA_ACC_GPU_CUDA_ENABLED
              -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND)
    # nvcc compilation flags
    target_compile_options(
      cuda.out PRIVATE --expt-relaxed-constexpr -gencode
                       arch=compute_61,code=[sm_61,compute_61])
    if(NOT "${ANNOTATE}" STREQUAL "")
      target_compile_options(cuda.out PRIVATE -DANNOTATE)
    endif()
    target_compile_options(cuda.out PRIVATE -pg)
    target_link_options(cuda.out PRIVATE -pg)
  endif()

  # # GPU HIP check if HIP is available check_language(HIP)
  # if(CMAKE_HIP_COMPILER) # enable HIP enable_language(HIP)
  # set(CMAKE_HIP_HOST_COMPILER ${CMAKE_HIP_COMPILER})

  # # look for the hip package folder find_package(hip)

  # set(hip_BASE "${hip_INCLUDE_DIRS}/..") # set the hipcc compiler
  # set(CMAKE_CXX_COMPILER "${hip_BASE}/bin/hipcc") # compile gpu hip module
  # add_executable(hip.out main.cpp) # link boost target_link_libraries(hip.out
  # PRIVATE ${Boost_LIBRARIES}) target_include_directories(hip.out PRIVATE
  # ${Boost_PATH}) # alpaka build flags target_compile_options( hip.out PRIVATE
  # -DALPAKA_ACC_GPU_HIP_PRESENT -DALPAKA_ACC_GPU_HIP_ENABLED
  # -DALPAKA_ACC_GPU_HIP_ASYNC_BACKEND) # link hip-rand libraries
  # target_include_directories(hip.out PRIVATE ${hip_INCLUDE_DIRS})
  # target_include_directories(hip.out PRIVATE ${hip_BASE}/hiprand/include)
  # target_include_directories(hip.out PRIVATE ${hip_BASE}/rocrand/include)
  # endif()
endif()
