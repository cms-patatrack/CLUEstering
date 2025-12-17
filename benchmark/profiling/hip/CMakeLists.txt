enable_language(HIP)
find_package(HIP)

set_source_files_properties(${CMAKE_SOURCE_DIR}/main.cpp PROPERTIES LANGUAGE
                                                                    HIP)
add_executable(hip.out ${CMAKE_SOURCE_DIR}/main.cpp)
target_include_directories(hip.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include)
target_link_libraries(hip.out PRIVATE alpaka::alpaka Boost::atomic)
target_compile_definitions(hip.out PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED
                                           CLUE_ENABLE_CACHING_ALLOCATOR)
