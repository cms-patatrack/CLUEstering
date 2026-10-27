
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "../Run.hpp"

#include <nanobind/nanobind.h>

namespace clue::alpaka_omp2_sync {

  NB_MODULE(CLUE_CPU_OMP, m) {
    m.doc() = "Binding of the CLUE algorithm running on CPU with OpenMP";

    m.def("listDevices",
          &listDevices,
          "List the available devices for the OpenMP backend");

    m.def("mainRun", &mainRun<float, clue::FlatKernel>);
    m.def("mainRun", &mainRun<float, clue::ExponentialKernel>);
    m.def("mainRun", &mainRun<float, clue::GaussianKernel>);

    m.def("mainRun", &mainRun<double, clue::FlatKernel>);
    m.def("mainRun", &mainRun<double, clue::ExponentialKernel>);
    m.def("mainRun", &mainRun<double, clue::GaussianKernel>);
  }
};  // namespace clue::alpaka_omp2_sync
