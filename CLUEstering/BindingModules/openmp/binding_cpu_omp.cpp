
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

namespace alpaka_omp2_async {

  PYBIND11_MODULE(CLUE_CPU_OMP, m) {
    m.doc() = "Binding of the CLUE algorithm running on CPU with OpenMP";

    m.def("listDevices",
          &alpaka_omp2_async::listDevices,
          "List the available devices for the OpenMP backend");
    m.def("mainRun", &alpaka_omp2_async::mainRun<float, clue::FlatKernel<float>>, "mainRun");
    m.def("mainRun", &alpaka_omp2_async::mainRun<float, clue::ExponentialKernel<float>>, "mainRun");
    m.def("mainRun", &alpaka_omp2_async::mainRun<float, clue::GaussianKernel<float>>, "mainRun");
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
    //           &alpaka_omp2_async::mainRun<double, clue::FlatKernel<double>>),
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
    //           &alpaka_omp2_async::mainRun<double, clue::ExponentialKernel<double>>),
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
    //           &alpaka_omp2_async::mainRun<double, clue::GaussianKernel<double>>),
    //       "mainRun");
  }
};  // namespace alpaka_omp2_async
