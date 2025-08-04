pybind11_add_module(CLUE_CPU_TBB SHARED binding_cpu_tbb.cpp)
target_include_directories(
  CLUE_CPU_TBB PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
                       ${alpaka_SOURCE_DIR}/include ${Boost_SOURCE_DIR})
target_compile_definitions(
  CLUE_CPU_TBB PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
                       CLUE_ENABLE_CACHING_ALLOCATOR)
target_link_libraries(CLUE_CPU_TBB PRIVATE TBB::tbb)
set_target_properties(
  CLUE_CPU_TBB
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_TBB
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_CPU_TBB.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
