
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

  // Parameter-free metrics
  m.def(
      "EuclideanMetric",
      []() { return Descriptor{Tag::Euclidean, {}}; },
      "Euclidean (L2) distance metric.");

  m.def(
      "ManhattanMetric",
      []() { return Descriptor{Tag::Manhattan, {}}; },
      "Manhattan (L1) distance metric.");

  m.def(
      "ChebyshevMetric",
      []() { return Descriptor{Tag::Chebyshev, {}}; },
      "Chebyshev (L-infinity) distance metric.");

  // Parameterised metrics
  m.def(
      "WeightedEuclideanMetric",
      [](std::vector<float> weights) {
        return Descriptor{Tag::WeightedEuclidean, std::move(weights)};
      },
      py::arg("weights"),
      "Weighted Euclidean metric. Pass one weight per coordinate dimension.");

  m.def(
      "WeightedChebyshevMetric",
      [](std::vector<float> weights) {
        return Descriptor{Tag::WeightedChebyshev, std::move(weights)};
      },
      py::arg("weights"),
      "Weighted Chebyshev metric. Pass one weight per coordinate dimension.");

  m.def(
      "PeriodicEuclideanMetric",
      [](std::vector<float> periods) {
        return Descriptor{Tag::PeriodicEuclidean, std::move(periods)};
      },
      py::arg("periods"),
      "Periodic Euclidean metric. Pass one period per coordinate dimension; "
      "a period of 0 means the dimension is not periodic.");
}
