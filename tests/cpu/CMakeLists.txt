file(GLOB sources ../*.cpp)

add_executable(serial.out ${sources})
target_include_directories(serial.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                              ${doctest_SOURCE_DIR}/doctest)
target_link_libraries(serial.out PRIVATE alpaka::alpaka Boost::boost fmt::fmt)
target_compile_definitions(
  serial.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
                     CLUE_ENABLE_CACHING_ALLOCATOR)
find_package(TBB)
if(TBB_FOUND)
  add_executable(tbb.out ${sources})
  target_include_directories(tbb.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                             ${doctest_SOURCE_DIR}/doctest)
  target_link_libraries(tbb.out PRIVATE alpaka::alpaka Boost::boost TBB::tbb
                                        fmt::fmt)
  target_compile_definitions(
    tbb.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
                    CLUE_ENABLE_CACHING_ALLOCATOR)
endif()

find_package(OpenMP)
# CPU OpenMP
if(OpenMP_CXX_FOUND)
  add_executable(openmp.out ${sources})
  target_include_directories(openmp.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                                ${doctest_SOURCE_DIR}/doctest)
  target_link_libraries(openmp.out PRIVATE alpaka::alpaka Boost::boost
                                           OpenMP::OpenMP_CXX fmt::fmt)
  target_compile_definitions(
    openmp.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                       CLUE_ENABLE_CACHING_ALLOCATOR)
endif()
if(COVERAGE)
  target_compile_options(openmp.out PRIVATE -O0 -g --coverage)
  target_link_options(openmp.out PRIVATE --coverage)
  target_compile_definitions(openmp.out PRIVATE COVERAGE)
endif()
