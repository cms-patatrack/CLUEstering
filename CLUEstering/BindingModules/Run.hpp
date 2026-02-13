
#pragma once

#include "CLUEstering/CLUEstering.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
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
               size_t block_size,
               size_t device_id) {
    auto rData = data.request();
    auto* pData = static_cast<TInput*>(rData.ptr);
    auto rResults = results.request();
    auto* pResults = static_cast<int*>(rResults.ptr);

    auto queue = clue::get_queue(device_id);

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1):
        run<TInput, 1, Kernel<TInput>>(dc,
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
        return;
      [[likely]] case (2):
        run<TInput, 2, Kernel<TInput>>(dc,
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
        return;
      [[likely]] case (3):
        run<TInput, 3, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] case (4):
        run<TInput, 4, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] case (5):
        run<TInput, 5, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] case (6):
        run<TInput, 6, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] case (7):
        run<TInput, 7, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] case (8):
        run<TInput, 8, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] case (9):
        run<TInput, 9, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] case (10):
        run<TInput, 10, Kernel<TInput>>(dc,
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
        return;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
    }
  }

}  // namespace ALPAKA_BACKEND
