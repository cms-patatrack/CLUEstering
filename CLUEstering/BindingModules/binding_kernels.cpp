
#include "CLUEstering/core/ConvolutionalKernel.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

PYBIND11_MODULE(CLUE_Convolutional_Kernels, m) {
  m.doc() = "Binding of the convolutional kernels used in the CLUE algorithm.";

  pybind11::class_<clue::FlatKernel<float>>(m, "FlatKernel").def(pybind11::init<float>());
  pybind11::class_<clue::ExponentialKernel<float>>(m, "ExponentialKernel")
      .def(pybind11::init<float, float>());
  pybind11::class_<clue::GaussianKernel<float>>(m, "GaussianKernel")
      .def(pybind11::init<float, float, float>());
}
