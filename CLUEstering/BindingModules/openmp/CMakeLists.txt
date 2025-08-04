pybind11_add_module(CLUE_CPU_OMP SHARED binding_cpu_omp.cpp)
target_include_directories(
  CLUE_CPU_OMP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
                       ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(
  CLUE_CPU_OMP PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                       CLUE_ENABLE_CACHING_ALLOCATOR)
target_link_libraries(CLUE_CPU_OMP PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(
  CLUE_CPU_OMP
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_OMP
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_CPU_OMP.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
