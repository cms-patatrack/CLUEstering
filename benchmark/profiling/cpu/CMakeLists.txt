# CPU Serial
add_executable(serial.out ${CMAKE_SOURCE_DIR}/main.cpp)
target_include_directories(
  serial.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../../include
          ${CMAKE_SOURCE_DIR}/../../extern/xtd/include
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(
  serial.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
                     CLUE_ENABLE_CACHING_ALLOCATOR)

find_package(TBB)
# CPU TBB
if(TBB_FOUND)
  add_executable(tbb.out ${CMAKE_SOURCE_DIR}/main.cpp)
  target_include_directories(
    tbb.out
    PRIVATE ${CMAKE_SOURCE_DIR}/../../include
            ${CMAKE_SOURCE_DIR}/../../extern/xtd/include
            ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
  target_link_libraries(tbb.out PRIVATE TBB::tbb)
  target_compile_definitions(
    tbb.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
                    CLUE_ENABLE_CACHING_ALLOCATOR)
endif()

find_package(OpenMP)
# CPU OpenMP
if(OpenMP_CXX_FOUND)
  add_executable(openmp.out ${CMAKE_SOURCE_DIR}/main.cpp)
  target_include_directories(
    openmp.out
    PRIVATE ${CMAKE_SOURCE_DIR}/../../include
            ${CMAKE_SOURCE_DIR}/../../extern/xtd/include
            ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
  target_link_libraries(openmp.out PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(
    openmp.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                       CLUE_ENABLE_CACHING_ALLOCATOR)
endif()
