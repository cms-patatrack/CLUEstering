
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
    m.def("mainRun", &alpaka_serial_sync::mainRun<float, clue::FlatKernel>);
    m.def("mainRun", &alpaka_serial_sync::mainRun<float, clue::ExponentialKernel>);
    m.def("mainRun", &alpaka_serial_sync::mainRun<float, clue::GaussianKernel>);

    m.def("mainRun", &alpaka_serial_sync::mainRun<double, clue::FlatKernel>);
    m.def("mainRun", &alpaka_serial_sync::mainRun<double, clue::ExponentialKernel>);
    m.def("mainRun", &alpaka_serial_sync::mainRun<double, clue::GaussianKernel>);
  }
};  // namespace alpaka_serial_sync
