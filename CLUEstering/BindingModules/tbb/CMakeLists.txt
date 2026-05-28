nanobind_add_module(CLUE_CPU_TBB NB_STATIC binding_cpu_tbb.cpp)
target_include_directories(CLUE_CPU_TBB
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_compile_definitions(
  CLUE_CPU_TBB PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
target_link_libraries(CLUE_CPU_TBB PRIVATE alpaka::alpaka Boost::atomic
                                           TBB::tbb)
target_precompile_headers(CLUE_CPU_TBB REUSE_FROM CLUE_Convolutional_Kernels)
set_target_properties(
  CLUE_CPU_TBB
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/)
install(TARGETS CLUE_CPU_TBB LIBRARY DESTINATION CLUEstering/lib)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_TBB
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CLUE_CPU_TBB>
          ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
