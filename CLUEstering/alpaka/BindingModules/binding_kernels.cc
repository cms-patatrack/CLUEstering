
#include <vector>

#include "../CLUE/ConvolutionalKernel.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

PYBIND11_MODULE(CLUE_Convolutional_Kernels, m) {
  m.doc() = "Binding of the convolutional kernels used in the CLUE algorithm.";

  pybind11::class_<ConvolutionalKernel>(m, "ConvolutionalKernel")
      .def(pybind11::init<kernel_t>())
      .def("operator()", &ConvolutionalKernel::operator());
}
