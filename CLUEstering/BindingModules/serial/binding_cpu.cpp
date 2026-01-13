
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "../Run.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace alpaka_serial_sync {

  PYBIND11_MODULE(CLUE_CPU_Serial, m) {
    m.doc() = "Binding of the CLUE algorithm running serially on CPU";

    m.def("listDevices",
          &alpaka_serial_sync::listDevices,
          "List the available devices for the CPU serial backend");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  float,
                                  int,
                                  std::vector<uint8_t>,
                                  py::array_t<float>,
                                  py::array_t<int>,
                                  const clue::FlatKernel&,
                                  int,
                                  int32_t,
                                  size_t,
                                  size_t>(&alpaka_serial_sync::mainRun<clue::FlatKernel>),
          "mainRun");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  float,
                                  int,
                                  std::vector<uint8_t>,
                                  py::array_t<float>,
                                  py::array_t<int>,
                                  const clue::ExponentialKernel&,
                                  int,
                                  int32_t,
                                  size_t,
                                  size_t>(&alpaka_serial_sync::mainRun<clue::ExponentialKernel>),
          "mainRun");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  float,
                                  int,
                                  std::vector<uint8_t>,
                                  py::array_t<float>,
                                  py::array_t<int>,
                                  const clue::GaussianKernel&,
                                  int,
                                  int32_t,
                                  size_t,
                                  size_t>(&alpaka_serial_sync::mainRun<clue::GaussianKernel>),
          "mainRun");
  }
};  // namespace alpaka_serial_sync
