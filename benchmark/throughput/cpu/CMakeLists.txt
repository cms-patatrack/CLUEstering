add_library(serial SHARED ${CMAKE_SOURCE_DIR}/worker.cpp)
target_compile_definitions(serial PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
target_include_directories(
  serial PRIVATE ${CMAKE_SOURCE_DIR}/../../include ${alpaka_SOURCE_DIR}/include
                 ${Boost_INCLUDE_DIR})
target_link_libraries(serial PRIVATE alpaka::alpaka Boost::boost)
