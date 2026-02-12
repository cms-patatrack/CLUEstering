file(GLOB sources ${CMAKE_SOURCE_DIR}/tests/*.cpp)

set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})

foreach(source ${sources})
  get_filename_component(test_name ${source} NAME_WE)
  set(target sycl_cpu_${test_name})

  add_executable(${target} ${sources})
  add_test(NAME ${target} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sycl/${test_name})
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                               ${doctest_SOURCE_DIR}/doctest)
  target_include_directories(${target} PRIVATE ${oneDPL_DIR})
  target_compile_options(${target} PRIVATE -DALPAKA_ACC_SYCL_ENABLED
                                           -DALPAKA_SYCL_ONEAPI_CPU -fsycl)
  target_compile_definitions(${target}
                             PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
  target_link_libraries(${target} PRIVATE alpaka::alpaka Boost::atomic fmt::fmt
                                          -fsycl)
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sycl/cpu
               OUTPUT_NAME ${test_name})
endforeach()

foreach(source ${sources})
  get_filename_component(test_name ${source} NAME_WE)
  set(target sycl_gpu_${test_name})

  add_executable(${target} ${sources})
  add_test(NAME ${target} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sycl/${test_name})
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                               ${doctest_SOURCE_DIR}/doctest)
  target_include_directories(${target} PRIVATE ${oneDPL_DIR})
  target_compile_options(${target} PRIVATE -DALPAKA_ACC_SYCL_ENABLED
                                           -DALPAKA_SYCL_ONEAPI_GPU -fsycl)
  target_compile_definitions(${target}
                             PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
  target_link_libraries(${target} PRIVATE alpaka::alpaka Boost::atomic fmt::fmt
                                          -fsycl)
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sycl/gpu
               OUTPUT_NAME ${test_name})
endforeach()
