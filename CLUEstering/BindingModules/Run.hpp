
#pragma once

#include "CLUEstering/CLUEstering.hpp"
#include "MetricDescriptor.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
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
         std::tuple<TInput*, int*>&& pData,
         const std::optional<std::span<uint32_t>>& batch_sample_sizes,
         int32_t n_points,
         const Kernel& kernel,
         const clue::internal::MetricDescriptor<TInput>& metric_desc,
         clue::Queue queue) {
  clue::Clusterer<Ndim, TInput> algo(queue, dc, rhoc, dm, seed_dc);
  algo.setWrappedCoordinates(std::move(wrapped));

  clue::PointsHost<Ndim, TInput> h_points(queue, n_points, std::get<0>(pData), std::get<1>(pData));
  clue::PointsDevice<Ndim, TInput> d_points(queue, n_points);

  clue::internal::apply_metric<Ndim>(metric_desc, [&](auto&& metric) {
    if (batch_sample_sizes.has_value()) [[unlikely]] {
      algo.make_clusters(queue, h_points, d_points, batch_sample_sizes.value(), metric, kernel);
    } else [[likely]] {
      algo.make_clusters(queue, h_points, d_points, metric, kernel);
    }
  });
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
               const clue::internal::MetricDescriptor<TInput>& metric_desc) {
    auto* pData = data.data();
    auto* pResults = results.data();

    std::optional<std::span<uint32_t>> batch_sample_sizes_span;

    if (batch_sample_sizes.has_value()) [[unlikely]] {
      auto* pBatchSizes = batch_sample_sizes->data();
      batch_sample_sizes_span = std::span<uint32_t>(pBatchSizes, batch_sample_sizes->size());
    } else [[likely]] {
      batch_sample_sizes_span = std::nullopt;
    }

    auto queue = clue::get_queue(device_id);
    auto dispatch = [&]<std::size_t N>() {
      run<TInput, N, Kernel<TInput>>(dc,
                                     rhoc,
                                     dm,
                                     seed_dc,
                                     std::move(wrapped),
                                     std::make_tuple(pData, pResults),
                                     batch_sample_sizes_span,
                                     n_points,
                                     kernel,
                                     metric_desc,
                                     queue);
    };
    switch (Ndim) {
      [[unlikely]] case (1):
        dispatch.template operator()<1>();
        return;
      [[likely]] case (2):
        dispatch.template operator()<2>();
        return;
      [[likely]] case (3):
        dispatch.template operator()<3>();
        return;
      [[unlikely]] case (4):
        dispatch.template operator()<4>();
        return;
      // [[unlikely]] case (5):
      //   dispatch.template operator()<5>();
      //   return;
      // [[unlikely]] case (6):
      //   dispatch.template operator()<6>();
      //   return;
      // [[unlikely]] case (7):
      //   dispatch.template operator()<7>();
      //   return;
      // [[unlikely]] case (8):
      //   dispatch.template operator()<8>();
      //   return;
      // [[unlikely]] case (9):
      //   dispatch.template operator()<9>();
      //   return;
      // [[unlikely]] case (10):
      //   dispatch.template operator()<10>();
        return;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
    }
  }

}  // namespace ALPAKA_BACKEND
