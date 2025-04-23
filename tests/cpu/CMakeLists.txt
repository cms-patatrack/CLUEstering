add_executable(serial.out ../test_clustering.cpp ../test_host_points.cpp ../test_device_points.cpp)
target_include_directories(
  serial.out
  PRIVATE ${CMAKE_SURCE_DIR}../../include/CLUEstering
          ${doctest_SOURCE_DIR}/doctest ${alpaka_SOURCE_DIR}/include
          ${Boost_INCLUDE_DIR})
target_compile_definitions(
  serial.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

find_package(TBB)
if(TBB_FOUND)
  add_executable(tbb.out ../test_clustering.cpp ../test_host_points.cpp ../test_device_points.cpp)
  target_include_directories(
    tbb.out
    PRIVATE ${CMAKE_SURCE_DIR}../../include/CLUEstering
            ${doctest_SOURCE_DIR}/doctest ${alpaka_SOURCE_DIR}/include
            ${Boost_INCLUDE_DIR})
  target_link_libraries(tbb.out PRIVATE TBB::tbb)
  target_compile_definitions(tbb.out PRIVATE ALPAKA_HOST_ONLY
                                             ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
endif()

find_package(OpenMP)
# CPU OpenMP
if(OpenMP_CXX_FOUND)
  add_executable(openmp.out ../test_clustering.cpp ../test_host_points.cpp ../test_device_points.cpp)
  target_include_directories(
    openmp.out
    PRIVATE ${CMAKE_SURCE_DIR}../../include/CLUEstering
            ${doctest_SOURCE_DIR}/doctest ${alpaka_SOURCE_DIR}/include
            ${Boost_INCLUDE_DIR})
  target_link_libraries(openmp.out PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(
    openmp.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
endif()
