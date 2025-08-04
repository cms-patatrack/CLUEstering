pybind11_add_module(CLUE_CPU_Serial SHARED binding_cpu.cpp)
target_include_directories(
  CLUE_CPU_Serial PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
                          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(
  CLUE_CPU_Serial PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
                          CLUE_ENABLE_CACHING_ALLOCATOR)
set_target_properties(
  CLUE_CPU_Serial
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_Serial
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_CPU_Serial.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
