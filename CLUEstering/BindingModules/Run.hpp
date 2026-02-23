
#pragma once

#include "CLUEstering/CLUEstering.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <span>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <std::floating_point TInput, std::size_t Ndim, clue::concepts::convolutional_kernel Kernel>
void run(TInput dc,
         TInput rhoc,
         TInput dm,
         TInput seed_dc,
         int pPBin,
         std::vector<uint8_t>&& wrapped,
         std::tuple<TInput*, int*>&& pData,
         int32_t n_points,
         const Kernel& kernel,
         clue::Queue queue,
         size_t block_size) {
  clue::Clusterer<Ndim, TInput> algo(queue, dc, rhoc, dm, seed_dc, pPBin);
  algo.setWrappedCoordinates(std::move(wrapped));

  clue::PointsHost<Ndim, TInput> h_points(queue, n_points, std::get<0>(pData), std::get<1>(pData));
  clue::PointsDevice<Ndim, TInput> d_points(queue, n_points);

  algo.make_clusters(
      queue, h_points, d_points, clue::EuclideanMetric<Ndim, TInput>{}, kernel, block_size);
}

template <std::floating_point TInput, std::size_t Ndim, clue::concepts::convolutional_kernel Kernel>
void run(TInput dc,
         TInput rhoc,
         TInput dm,
         TInput seed_dc,
         int pPBin,
         std::vector<uint8_t>&& wrapped,
         std::tuple<TInput*, int*>&& pData,
         int32_t n_points,
         const Kernel& kernel,
         std::span<uint32_t> batch_sample_sizes,
         clue::Queue queue,
         size_t block_size) {
  clue::Clusterer<Ndim, TInput> algo(queue, dc, rhoc, dm, seed_dc, pPBin);
  algo.setWrappedCoordinates(std::move(wrapped));

  clue::PointsHost<Ndim, TInput> h_points(queue, n_points, std::get<0>(pData), std::get<1>(pData));
  clue::PointsDevice<Ndim, TInput> d_points(queue, n_points);

  algo.make_clusters(queue,
                     h_points,
                     d_points,
                     batch_sample_sizes,
                     clue::EuclideanMetric<Ndim, TInput>{},
                     kernel,
                     block_size);
}

namespace ALPAKA_BACKEND {

  void listDevices(const std::string& backend) {
    const char tab = '\t';
    const std::vector<Device> devices = alpaka::getDevs(clue::Platform{});
    if (devices.empty()) {
      std::cout << "No devices found for the " << backend << " backend." << std::endl;
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
               int pPBin,
               std::vector<uint8_t> wrapped,
               py::array_t<TInput> data,
               py::array_t<int> results,
               const Kernel<TInput>& kernel,
               int Ndim,
               int32_t n_points,
               std::size_t block_size,
               std::size_t device_id) {
    auto rData = data.request();
    auto* pData = static_cast<TInput*>(rData.ptr);
    auto rResults = results.request();
    auto* pResults = static_cast<int*>(rResults.ptr);

    auto queue = clue::get_queue(device_id);

    auto dispatch = [&]<std::size_t N>() {
      run<TInput, N, Kernel<TInput>>(dc,
                                     rhoc,
                                     dm,
                                     seed_dc,
                                     pPBin,
                                     std::move(wrapped),
                                     std::make_tuple(pData, pResults),
                                     n_points,
                                     kernel,
                                     queue,
                                     block_size);
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
      [[unlikely]] case (5):
        dispatch.template operator()<5>();
        return;
      [[unlikely]] case (6):
        dispatch.template operator()<6>();
        return;
      [[unlikely]] case (7):
        dispatch.template operator()<7>();
        return;
      [[unlikely]] case (8):
        dispatch.template operator()<8>();
        return;
      [[unlikely]] case (9):
        dispatch.template operator()<9>();
        return;
      [[unlikely]] case (10):
        dispatch.template operator()<10>();
        return;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
    }
  }

  template <std::floating_point TInput, template <typename T> typename Kernel>
    requires clue::concepts::convolutional_kernel<Kernel<TInput>>
  void mainRun(TInput dc,
               TInput rhoc,
               TInput dm,
               TInput seed_dc,
               int pPBin,
               std::vector<uint8_t> wrapped,
               py::array_t<TInput> data,
               py::array_t<int> results,
               const Kernel<TInput>& kernel,
               int Ndim,
               py::array_t<uint32_t> batch_sample_sizes,
               int32_t n_points,
               std::size_t block_size,
               std::size_t device_id) {
    auto rData = data.request();
    auto* pData = static_cast<TInput*>(rData.ptr);
    auto rResults = results.request();
    auto* pResults = static_cast<int*>(rResults.ptr);

    auto rBatchSizes = batch_sample_sizes.request();
    auto* pBatchSizes = static_cast<uint32_t*>(rBatchSizes.ptr);
    std::span<uint32_t> batch_sizes(pBatchSizes, rBatchSizes.size);

    auto queue = clue::get_queue(device_id);

    auto dispatch = [&]<std::size_t N>() {
      run<TInput, N, Kernel<TInput>>(dc,
                                     rhoc,
                                     dm,
                                     seed_dc,
                                     pPBin,
                                     std::move(wrapped),
                                     std::make_tuple(pData, pResults),
                                     batch_sizes,
                                     n_points,
                                     kernel,
                                     queue,
                                     block_size);
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
      [[unlikely]] case (5):
        dispatch.template operator()<5>();
        return;
      [[unlikely]] case (6):
        dispatch.template operator()<6>();
        return;
      [[unlikely]] case (7):
        dispatch.template operator()<7>();
        return;
      [[unlikely]] case (8):
        dispatch.template operator()<8>();
        return;
      [[unlikely]] case (9):
        dispatch.template operator()<9>();
        return;
      [[unlikely]] case (10):
        dispatch.template operator()<10>();
        return;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
    }
  }

}  // namespace ALPAKA_BACKEND
