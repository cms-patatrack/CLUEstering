
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "MetricDescriptor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

PYBIND11_MODULE(CLUE_Convolutional_Kernels, m) {
  m.doc() = "Binding of the convolutional kernels and distance metrics used in the CLUE algorithm.";

  // ----- Convolutional kernels -----
  py::class_<clue::FlatKernel<float>>(m, "FlatKernel").def(py::init<float>());
  py::class_<clue::ExponentialKernel<float>>(m, "ExponentialKernel").def(py::init<float, float>());
  py::class_<clue::GaussianKernel<float>>(m, "GaussianKernel").def(py::init<float, float, float>());

  // ----- Distance metrics -----
  // MetricDescriptor<float> is an opaque handle; users obtain instances via the
  // factory functions below rather than constructing it directly.
  using Descriptor = clue::internal::MetricDescriptor<float>;
  using Tag = Descriptor::Tag;

  py::class_<Descriptor>(m, "MetricDescriptor");

  m.def(
      "EuclideanMetric",
      [](std::vector<float> weights) {
        return Descriptor{Tag::WeightedEuclidean, std::move(weights)};
      },
      py::arg("weights") = std::vector<float>{},
      "Euclidean (L2) distance metric. Optionally pass one weight per coordinate dimension; "
      "omit or pass an empty list for the unweighted variant.");

  m.def(
      "ManhattanMetric",
      []() { return Descriptor{Tag::Manhattan, {}}; },
      "Manhattan (L1) distance metric.");

  m.def(
      "ChebyshevMetric",
      [](std::vector<float> weights) {
        return Descriptor{Tag::WeightedChebyshev, std::move(weights)};
      },
      py::arg("weights") = std::vector<float>{},
      "Chebyshev (L-infinity) distance metric. Optionally pass one weight per coordinate "
      "dimension; omit or pass an empty list for the unweighted variant.");

  m.def(
      "PeriodicEuclideanMetric",
      [](std::vector<float> periods) {
        return Descriptor{Tag::PeriodicEuclidean, std::move(periods)};
      },
      py::arg("periods"),
      "Periodic Euclidean metric. Pass one period per coordinate dimension; "
      "a period of 0 means the dimension is not periodic.");
}
