
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

namespace alpaka_rocm_async {

  PYBIND11_MODULE(CLUE_GPU_HIP, m) {
    m.doc() = "Binding of the CLUE algorithm running on AMD GPUs";

    m.def("listDevices",
          &alpaka_rocm_async::listDevices,
          "List the available devices for the HIP/ROCm backend");
    m.def("mainRun", &alpaka_rocm_async::mainRun<float, clue::FlatKernel>, "mainRun");
    m.def("mainRun", &alpaka_rocm_async::mainRun<float, clue::ExponentialKernel>, "mainRun");
    m.def("mainRun", &alpaka_rocm_async::mainRun<float, clue::GaussianKernel>, "mainRun");
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
    //           &alpaka_rocm_async::mainRun<double, clue::FlatKernel<double>>),
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
    //           &alpaka_rocm_async::mainRun<double, clue::ExponentialKernel<double>>),
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
    //           &alpaka_rocm_async::mainRun<double, clue::GaussianKernel<double>>),
    //       "mainRun");
  }
};  // namespace alpaka_rocm_async
