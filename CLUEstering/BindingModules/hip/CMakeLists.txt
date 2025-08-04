enable_language(HIP)
set(CMAKE_HIP_HOST_COMPILER ${CMAKE_HIP_COMPILER})

find_package(hip)

set(hip_BASE "${hip_INCLUDE_DIRS}/..")
set(CMAKE_CXX_COMPILER "${hip_BASE}/bin/hipcc")
pybind11_add_module(CLUE_GPU_HIP SHARED binding_gpu_hip.cpp)
target_include_directories(
  CLUE_GPU_HIP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
                       ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(CLUE_GPU_HIP PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED
                                                CLUE_ENABLE_CACHING_ALLOCATOR)
target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_INCLUDE_DIRS})
target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_BASE}/hiprand/include)
target_include_directories(CLUE_GPU_HIP PRIVATE ${hip_BASE}/rocrand/include)
set_target_properties(
  CLUE_GPU_HIP
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/)
# create link of shared object to lib folder inside CLUEstering directory
add_custom_command(
  TARGET CLUE_GPU_HIP
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_GPU_HIP.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
