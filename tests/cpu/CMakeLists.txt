file(GLOB sources ../*.cpp)

foreach(source ${sources})
  get_filename_component(test_name ${source} NAME_WE)
  set(target serial_${test_name})

  add_executable(${target} ${source})
  add_test(NAME ${target} COMMAND ${CMAKE_BINARY_DIR}/serial/${test_name})
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                               ${doctest_SOURCE_DIR}/doctest)
  target_link_libraries(${target} PRIVATE alpaka::alpaka Boost::atomic fmt::fmt)
  target_compile_definitions(
    ${target} PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
                      CLUE_ENABLE_CACHING_ALLOCATOR)
  set_target_properties(
    ${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/serial
                         OUTPUT_NAME ${test_name})
  if(COVERAGE)
    target_compile_options(${target} PRIVATE -O0 -g --coverage)
    target_link_options(${target} PRIVATE --coverage)
    target_compile_definitions(${target} PRIVATE COVERAGE)
  endif()
endforeach()

find_package(TBB)
if(TBB_FOUND AND NOT COVERAGE)
  foreach(source ${sources})
    get_filename_component(test_name ${source} NAME_WE)
    set(target tbb_${test_name})

    add_executable(${target} ${source})
    add_test(NAME ${target} COMMAND ${CMAKE_BINARY_DIR}/tbb/${test_name})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                                 ${doctest_SOURCE_DIR}/doctest)
    target_link_libraries(${target} PRIVATE alpaka::alpaka Boost::atomic
                                            TBB::tbb fmt::fmt)
    target_compile_definitions(
      ${target} PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
                        CLUE_ENABLE_CACHING_ALLOCATOR)
    set_target_properties(
      ${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tbb
                           OUTPUT_NAME ${test_name})
  endforeach()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND AND NOT COVERAGE)
  foreach(source ${sources})
    get_filename_component(test_name ${source} NAME_WE)
    set(target openmp_${test_name})
    add_executable(${target} ${source})
    add_test(NAME ${target} COMMAND ${CMAKE_BINARY_DIR}/openmp/${test_name})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                                 ${doctest_SOURCE_DIR}/doctest)
    target_link_libraries(${target} PRIVATE alpaka::alpaka Boost::atomic
                                            OpenMP::OpenMP_CXX fmt::fmt)
    target_compile_definitions(
      ${target} PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                        CLUE_ENABLE_CACHING_ALLOCATOR)
    set_target_properties(
      ${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/openmp
                           OUTPUT_NAME ${test_name})
  endforeach()
endif()
