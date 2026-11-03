
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "../Run.hpp"

#include <nanobind/nanobind.h>

namespace clue::alpaka_rocm_async {

  inline std::size_t deviceCount() { return alpaka::getDevCount(clue::Platform{}); }

  NB_MODULE(CLUE_GPU_HIP, m) {
    m.doc() = "Binding of the CLUE algorithm running on AMD GPUs";

    m.def("listDevices", &listDevices, "List the available devices for the HIP/ROCm backend");
    m.def("_deviceCount",
          &deviceCount,
          "Get the number of available devices for the HIP/ROCm backend");

    m.def("mainRun", &mainRun<float, clue::FlatKernel>);
    m.def("mainRun", &mainRun<float, clue::ExponentialKernel>);
    m.def("mainRun", &mainRun<float, clue::GaussianKernel>);

    m.def("mainRun", &mainRun<double, clue::FlatKernel>);
    m.def("mainRun", &mainRun<double, clue::ExponentialKernel>);
    m.def("mainRun", &mainRun<double, clue::GaussianKernel>);
  }
};  // namespace clue::alpaka_rocm_async
