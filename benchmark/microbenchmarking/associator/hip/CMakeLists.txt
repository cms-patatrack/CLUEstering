enable_language(HIP)
find_package(HIP)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set_source_files_properties(${CMAKE_SOURCE_DIR}/associator_build.cpp
                            PROPERTIES LANGUAGE HIP)
add_executable(hip.out ${CMAKE_SOURCE_DIR}/associator_build.cpp)
target_include_directories(hip.out PRIVATE ${CMAKE_SOURCE_DIR}/../../../include)
target_link_libraries(hip.out PRIVATE alpaka::alpaka Boost::boost
                                      benchmark::benchmark)
target_compile_definitions(hip.out PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED
                                           CLUE_ENABLE_CACHING_ALLOCATOR)
