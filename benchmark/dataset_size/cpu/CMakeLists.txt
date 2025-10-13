add_executable(serial.out ${CMAKE_SOURCE_DIR}/main.cpp)
target_include_directories(
  serial.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                     ${CMAKE_SOURCE_DIR}/../../benchmark)
target_link_libraries(serial.out PRIVATE alpaka::alpaka Boost::boost
                                         ${pybind_link})
target_compile_definitions(
  serial.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
                     CLUE_ENABLE_CACHING_ALLOCATOR)

add_executable(threads.out ${CMAKE_SOURCE_DIR}/main.cpp)
target_include_directories(
  threads.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                      ${CMAKE_SOURCE_DIR}/../../benchmark)
target_link_libraries(threads.out PRIVATE alpaka::alpaka Boost::boost
                                          ${pybind_link})
target_compile_definitions(
  threads.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
                      CLUE_ENABLE_CACHING_ALLOCATOR)

find_package(TBB)
if(TBB_FOUND)
  add_executable(tbb.out ${CMAKE_SOURCE_DIR}/main.cpp)
  target_include_directories(
    tbb.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                    ${CMAKE_SOURCE_DIR}/../../benchmark)
  target_link_libraries(tbb.out PRIVATE alpaka::alpaka Boost::boost TBB::tbb
                                        ${pybind_link})
  target_compile_definitions(
    tbb.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
                    CLUE_ENABLE_CACHING_ALLOCATOR)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_executable(openmp.out ${CMAKE_SOURCE_DIR}/main.cpp)
  target_include_directories(
    openmp.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                       ${CMAKE_SOURCE_DIR}/../../benchmark)
  target_link_libraries(openmp.out PRIVATE alpaka::alpaka Boost::boost
                                           OpenMP::OpenMP_CXX ${pybind_link})
  target_compile_definitions(
    openmp.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                       CLUE_ENABLE_CACHING_ALLOCATOR)
  set_target_properties(openmp.out PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              ${CMAKE_BINARY_DIR})
endif()
