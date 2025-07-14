cmake_minimum_required(VERSION 3.16.0)
project(
  CLUEstering
  LANGUAGES CXX
  VERSION 2.6.4.3)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(
  APPEND
  CMAKE_CXX_FLAGS_DEBUG
  " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -Wall -Wextra -Wpedantic -Wpedantic -Werror"
)
string(
  APPEND
  CMAKE_CXX_FLAGS
  "-O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

# include pybind11 extern subfolder
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_VERSION ">=3.8")
add_subdirectory(extern/pybind11)

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  alpaka
  URL https://github.com/alpaka-group/alpaka/archive/refs/tags/1.2.0.tar.gz)
FetchContent_MakeAvailable(alpaka)

# create lib directory in CLUEstering folder
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CLUEstering/lib)

add_subdirectory(CLUEstering/BindingModules)

# generate version header
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/CLUEstering/version.hpp @ONLY)

set(CLUE_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/CLUEstering")

add_library(CLUEstering INTERFACE)
target_include_directories(
  CLUEstering
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/xtd/include>
            $<BUILD_INTERFACE:${CLUE_GENERATED_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CLUEstering/version.hpp
        DESTINATION include/CLUEstering)
install(TARGETS CLUEstering EXPORT CLUEsteringTargets)
install(
  EXPORT CLUEsteringTargets
  FILE CLUEsteringTargets.cmake
  NAMESPACE clue::
  DESTINATION lib/cmake/CLUEstering)

set(CLUEstering_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CLUEstering_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CLUEstering_VERSION_PATCH ${PROJECT_VERSION_PATCH})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CLUEsteringConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfig.cmake"
  INSTALL_DESTINATION lib/cmake/CLUEstering)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfigVersion.cmake"
        DESTINATION lib/cmake/CLUEstering)
