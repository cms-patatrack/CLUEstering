
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
    m.def("mainRun", &alpaka_serial_sync::mainRun<float, clue::FlatKernel<float>>, "mainRun");
    m.def(
        "mainRun", &alpaka_serial_sync::mainRun<float, clue::ExponentialKernel<float>>, "mainRun");
    m.def("mainRun", &alpaka_serial_sync::mainRun<float, clue::GaussianKernel<float>>, "mainRun");
    // m.def("mainRun",
    //       pybind11::overload_cast<double,
    //                               double,
    //                               double,
    //                               double,
    //                               int,
    //                               std::vector<uint8_t>,
    //                               py::array_t<double>,
    //                               py::array_t<int>,
    //                               const clue::FlatKernel<double>&,
    //                               int,
    //                               int32_t,
    //                               size_t,
    //                               size_t>(
    //           &alpaka_serial_sync::mainRun<double, clue::FlatKernel<double>>),
    //       "mainRun");
    // m.def("mainRun",
    //       pybind11::overload_cast<double,
    //                               double,
    //                               double,
    //                               double,
    //                               int,
    //                               std::vector<uint8_t>,
    //                               py::array_t<double>,
    //                               py::array_t<int>,
    //                               const clue::ExponentialKernel<double>&,
    //                               int,
    //                               int32_t,
    //                               size_t,
    //                               size_t>(
    //           &alpaka_serial_sync::mainRun<double, clue::ExponentialKernel<double>>),
    //       "mainRun");
    // m.def("mainRun",
    //       pybind11::overload_cast<double,
    //                               double,
    //                               double,
    //                               double,
    //                               int,
    //                               std::vector<uint8_t>,
    //                               py::array_t<double>,
    //                               py::array_t<int>,
    //                               const clue::GaussianKernel<double>&,
    //                               int,
    //                               int32_t,
    //                               size_t,
    //                               size_t>(
    //           &alpaka_serial_sync::mainRun<double, clue::GaussianKernel<double>>),
    //       "mainRun");
  }
};  // namespace alpaka_serial_sync
