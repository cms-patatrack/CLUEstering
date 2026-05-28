
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "../Run.hpp"

#include <nanobind/nanobind.h>

namespace alpaka_serial_sync {

  NB_MODULE(CLUE_CPU_Serial, m) {
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
