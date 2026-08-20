
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "../Run.hpp"

#include <nanobind/nanobind.h>

namespace alpaka_cuda_async {

  NB_MODULE(CLUE_GPU_CUDA, m) {
    m.doc() = "Binding of the CLUE algorithm running on CUDA GPUs";

    m.def("listDevices",
          &alpaka_cuda_async::listDevices,
          "List the available devices for the CUDA backend");

    m.def("mainRun", &alpaka_cuda_async::mainRun<float, clue::FlatKernel>);
    m.def("mainRun", &alpaka_cuda_async::mainRun<float, clue::ExponentialKernel>);
    m.def("mainRun", &alpaka_cuda_async::mainRun<float, clue::GaussianKernel>);

    m.def("mainRun", &alpaka_cuda_async::mainRun<double, clue::FlatKernel>);
    m.def("mainRun", &alpaka_cuda_async::mainRun<double, clue::ExponentialKernel>);
    m.def("mainRun", &alpaka_cuda_async::mainRun<double, clue::GaussianKernel>);
  }
};  // namespace alpaka_cuda_async
