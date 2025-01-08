
#include "../CLUE/ConvolutionalKernel.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

PYBIND11_MODULE(CLUE_Convolutional_Kernels, m) {
  m.doc() = "Binding of the convolutional kernels used in the CLUE algorithm.";

  pybind11::class_<FlatKernel>(m, "FlatKernel").def(pybind11::init<float>());
  pybind11::class_<ExponentialKernel>(m, "ExponentialKernel")
      .def(pybind11::init<float, float>());
  pybind11::class_<GaussianKernel>(m, "GaussianKernel")
      .def(pybind11::init<float, float, float>());
}
