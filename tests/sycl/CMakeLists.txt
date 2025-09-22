file(GLOB sources ../*.cpp)

set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})
add_executable(sycl_cpu.out ${sources})
target_include_directories(
  sycl_cpu.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../include ${doctest_SOURCE_DIR}/doctest
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_include_directories(sycl_cpu.out PRIVATE ${oneDPL_DIR})
target_compile_options(sycl_cpu.out PRIVATE -DALPAKA_ACC_SYCL_ENABLED
                                            -DALPAKA_SYCL_ONEAPI_CPU -fsycl)
target_link_libraries(sycl_cpu.out PRIVATE fmt::fmt -fsycl)

add_executable(sycl_gpu.out ${sources})
target_include_directories(
  sycl_gpu.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../include ${doctest_SOURCE_DIR}/doctest
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_include_directories(sycl_gpu.out PRIVATE ${oneDPL_DIR})
target_compile_options(sycl_gpu.out PRIVATE -DALPAKA_ACC_SYCL_ENABLED
                                            -DALPAKA_SYCL_ONEAPI_GPU -fsycl)
target_link_libraries(sycl_gpu.out PRIVATE fmt::fmt -fsycl)
