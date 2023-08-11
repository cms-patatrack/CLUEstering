
#include <vector>

#include "../CLUE/ConvolutionalKernel.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

PYBIND11_MODULE(CLUE_Convolutional_Kernels, m) {
  m.doc() = "Binding of the convolutional kernels used in the CLUE algorithm.";

  pybind11::class_<ConvolutionalKernel>(m, "ConvolutionalKernel")
      .def(pybind11::init<>())
      .def("operator()", &ConvolutionalKernel::operator());
  pybind11::class_<FlatKernel, ConvolutionalKernel>(m, "FlatKernel")
      .def(pybind11::init<float>())
      .def("operator()", &FlatKernel::operator());
  pybind11::class_<GaussianKernel, ConvolutionalKernel>(m, "GaussianKernel")
      .def(pybind11::init<float, float, float>())
      .def("operator()", &GaussianKernel::operator());
  pybind11::class_<ExponentialKernel, ConvolutionalKernel>(m, "ExponentialKernel")
      .def(pybind11::init<float, float>())
      .def("operator()", &ExponentialKernel::operator());
  pybind11::class_<CustomKernel, ConvolutionalKernel>(m, "CustomKernel")
      .def(pybind11::init<kernel_t>())
      .def("operator()", &CustomKernel::operator());
}
