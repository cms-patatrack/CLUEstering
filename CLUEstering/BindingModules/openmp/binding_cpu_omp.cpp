
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "../Run.hpp"

#include <nanobind/nanobind.h>

namespace alpaka_omp2_async {

  NB_MODULE(CLUE_CPU_OMP, m) {
    m.doc() = "Binding of the CLUE algorithm running on CPU with OpenMP";

    m.def("listDevices",
          &alpaka_omp2_async::listDevices,
          "List the available devices for the OpenMP backend");

    m.def("mainRun", &alpaka_omp2_async::mainRun<float, clue::FlatKernel>);
    m.def("mainRun", &alpaka_omp2_async::mainRun<float, clue::ExponentialKernel>);
    m.def("mainRun", &alpaka_omp2_async::mainRun<float, clue::GaussianKernel>);

    m.def("mainRun", &alpaka_omp2_async::mainRun<double, clue::FlatKernel>);
    m.def("mainRun", &alpaka_omp2_async::mainRun<double, clue::ExponentialKernel>);
    m.def("mainRun", &alpaka_omp2_async::mainRun<double, clue::GaussianKernel>);
  }
};  // namespace alpaka_omp2_async
