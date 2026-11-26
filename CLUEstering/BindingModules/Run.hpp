
#pragma once

#include "CLUEstering/CLUEstering.hpp"
#include "MetricDescriptor.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <span>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

template <std::floating_point TInput, std::size_t Ndim, clue::concepts::convolutional_kernel Kernel>
void run(TInput dc,
         TInput rhoc,
         TInput dm,
         TInput seed_dc,
         std::vector<uint8_t>&& wrapped,
         clue::PointsHost<Ndim, TInput>& h_points,
         const std::optional<std::span<uint32_t>>& batch_sample_sizes,
         const Kernel& kernel,
         const clue::bindings::MetricDescriptor<TInput>& metric_desc,
         std::optional<TInput> merge_threshold,
         clue::Queue& queue) {
  clue::Clusterer<Ndim, TInput> algo(dc, rhoc, dm, seed_dc);
  algo.setWrappedCoordinates(std::move(wrapped));
  algo.setMergeThreshold(merge_threshold);

  clue::PointsDevice<Ndim, TInput> d_points(queue, h_points.size());

  clue::internal::apply_metric<Ndim>(metric_desc, [&](auto&& metric) {
    if (batch_sample_sizes.has_value()) [[unlikely]] {
      algo.make_clusters(queue, h_points, d_points, batch_sample_sizes.value(), metric, kernel);
    } else [[likely]] {
      algo.make_clusters(queue, h_points, d_points, metric, kernel);
    }
  });
}

// Calls f.template operator()<N>() with N equal to ndim, for the dimensions compiled in the
// bindings. Returns false if ndim is not supported.
template <typename F>
bool dispatch_ndim(int ndim, F&& f) {
  switch (ndim) {
    [[unlikely]] case (1):
      f.template operator()<1>();
      return true;
    [[likely]] case (2):
      f.template operator()<2>();
      return true;
    [[likely]] case (3):
      f.template operator()<3>();
      return true;
    [[unlikely]] case (4):
      f.template operator()<4>();
      return true;
    [[unlikely]] default:
      return false;
  }
}

inline std::optional<std::span<uint32_t>> as_span(
    std::optional<nb::ndarray<uint32_t, nb::numpy>>& batch_sample_sizes) {
  if (batch_sample_sizes.has_value()) [[unlikely]] {
    return std::span<uint32_t>(batch_sample_sizes->data(), batch_sample_sizes->size());
  }
  return std::nullopt;
}

namespace ALPAKA_BACKEND {

  inline void listDevices(const std::string& backend) {
    const char tab = '\t';
    const std::vector<Device> devices = alpaka::getDevs(clue::Platform{});
    if (devices.empty()) {
      std::cout << "No devices found for the " << backend << " backend.\n";
      return;
    } else {
      std::cout << backend << " devices found: \n";
      for (auto i = 0u; i < devices.size(); ++i) {
        std::cout << tab << "device " << i << ": " << alpaka::getName(devices[i]) << '\n';
      }
    }
  }

  template <std::floating_point TInput, template <typename T> typename Kernel>
    requires clue::concepts::convolutional_kernel<Kernel<TInput>>
  void mainRun(TInput dc,
               TInput rhoc,
               TInput dm,
               TInput seed_dc,
               std::vector<uint8_t> wrapped,
               nb::ndarray<TInput, nb::numpy> data,
               nb::ndarray<int, nb::numpy> results,
               const Kernel<TInput>& kernel,
               int Ndim,
               std::optional<nb::ndarray<uint32_t, nb::numpy>> batch_sample_sizes,
               int32_t n_points,
               std::size_t device_id,
               const clue::bindings::MetricDescriptor<TInput>& metric_desc,
               std::optional<TInput> merge_threshold) {
    auto* pData = data.data();
    auto* pResults = results.data();
    const auto batch_sample_sizes_span = as_span(batch_sample_sizes);

    auto queue = clue::get_queue(device_id);
    const auto supported = dispatch_ndim(Ndim, [&]<std::size_t N>() {
      clue::PointsHost<N, TInput> h_points(queue, n_points, pData, pResults);
      run<TInput, N, Kernel<TInput>>(dc,
                                     rhoc,
                                     dm,
                                     seed_dc,
                                     std::move(wrapped),
                                     h_points,
                                     batch_sample_sizes_span,
                                     kernel,
                                     metric_desc,
                                     merge_threshold,
                                     queue);
    });
    if (!supported) [[unlikely]] {
      std::cout << "The Python interface only supports up to 4 dimensions\n";
    }
  }

  // Same as mainRun, with the coordinates and the weights in separate arrays, used directly as
  // the host points without copies. It also returns the density, the distance to the nearest
  // higher (+inf for the points without one) and the nearest higher of each point. The
  // coordinates have shape (n_points, Ndim) and must be Fortran-contiguous, which is the structure
  // of arrays layout of PointsHost.
  template <std::floating_point TInput, template <typename T> typename Kernel>
    requires clue::concepts::convolutional_kernel<Kernel<TInput>>
  void mainRunSeparate(TInput dc,
                       TInput rhoc,
                       TInput dm,
                       TInput seed_dc,
                       std::vector<uint8_t> wrapped,
                       nb::ndarray<TInput, nb::numpy, nb::ndim<2>, nb::f_contig> coordinates,
                       nb::ndarray<TInput, nb::numpy, nb::ndim<1>, nb::c_contig> weights,
                       nb::ndarray<int, nb::numpy, nb::ndim<1>, nb::c_contig> results,
                       nb::ndarray<TInput, nb::numpy, nb::ndim<1>, nb::c_contig> rho,
                       nb::ndarray<TInput, nb::numpy, nb::ndim<1>, nb::c_contig> delta,
                       nb::ndarray<int, nb::numpy, nb::ndim<1>, nb::c_contig> nearest_higher,
                       const Kernel<TInput>& kernel,
                       std::optional<nb::ndarray<uint32_t, nb::numpy>> batch_sample_sizes,
                       std::size_t device_id,
                       const clue::bindings::MetricDescriptor<TInput>& metric_desc,
                       std::optional<TInput> merge_threshold) {
    const auto n_points = coordinates.shape(0);
    const auto ndim = static_cast<int>(coordinates.shape(1));
    if (weights.shape(0) != n_points || results.shape(0) != n_points || rho.shape(0) != n_points ||
        delta.shape(0) != n_points || nearest_higher.shape(0) != n_points) {
      throw std::invalid_argument(
          "The coordinates, the weights and the outputs must have the same number of points");
    }
    const auto batch_sample_sizes_span = as_span(batch_sample_sizes);

    auto queue = clue::get_queue(device_id);
    const auto supported = dispatch_ndim(ndim, [&]<std::size_t N>() {
      clue::PointsHost<N, TInput> h_points(
          queue, static_cast<int32_t>(n_points), coordinates.data(), weights.data(), results.data());
      h_points.store_details();
      run<TInput, N, Kernel<TInput>>(dc,
                                     rhoc,
                                     dm,
                                     seed_dc,
                                     std::move(wrapped),
                                     h_points,
                                     batch_sample_sizes_span,
                                     kernel,
                                     metric_desc,
                                     merge_threshold,
                                     queue);
      // TODO: let store_details use the output arrays directly, to avoid these copies
      std::ranges::copy(h_points.rho(), rho.data());
      std::ranges::copy(h_points.delta(), delta.data());
      std::ranges::copy(h_points.nearest_higher(), nearest_higher.data());
    });
    if (!supported) [[unlikely]] {
      throw std::invalid_argument("The Python interface only supports up to 4 dimensions");
    }
  }

  // Registers mainRunSeparate for the kernels. Only float is registered, because the kernels and
  // the metrics are bound only for float. The outputs are written in place, so they must not be
  // converted, which would write them into a temporary copy.
  inline void defMainRunSeparate(nb::module_& m) {
    auto def = [&]<typename TInput, template <typename T> typename Kernel>() {
      m.def("mainRunSeparate",
            &mainRunSeparate<TInput, Kernel>,
            nb::arg("dc"),
            nb::arg("rhoc"),
            nb::arg("dm"),
            nb::arg("seed_dc"),
            nb::arg("wrapped"),
            nb::arg("coordinates"),
            nb::arg("weights"),
            nb::arg("results").noconvert(),
            nb::arg("rho").noconvert(),
            nb::arg("delta").noconvert(),
            nb::arg("nearest_higher").noconvert(),
            nb::arg("kernel"),
            nb::arg("batch_sample_sizes").none(),
            nb::arg("device_id"),
            nb::arg("metric"),
            nb::arg("merge_threshold").none());
    };
    def.template operator()<float, clue::FlatKernel>();
    def.template operator()<float, clue::ExponentialKernel>();
    def.template operator()<float, clue::GaussianKernel>();
  }

}  // namespace ALPAKA_BACKEND
