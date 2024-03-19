#include <alpaka/alpaka.hpp>
#include <vector>

#include "../CLUE/CLUEAlgoAlpaka.h"
#include "../CLUE/Run.h"
#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "../AlpakaCore/initialise.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

using cms::alpakatools::initialise;

namespace alpaka_cuda_async {
  void listDevices(const std::string& backend) {
    const char tab = '\t';
    const std::vector<Device> devices = alpaka::getDevs<Platform>();
    if (devices.empty()) {
      std::cout << "No devices found for the " << backend << " backend." << std::endl;
      return;
    } else {
      std::cout << backend << " devices found: \n";
      for (size_t i{}; i < devices.size(); ++i) {
        std::cout << tab << "device " << i << ": " << alpaka::getName(devices[i]) << '\n';
      }
    }
  }

  std::vector<std::vector<int>> mainRun(float dc,
                                        float rhoc,
                                        float outlier,
                                        int pPBin,
                                        const std::vector<std::vector<float>>& coords,
                                        const std::vector<float>& weights,
                                        const FlatKernel& kernel,
                                        int Ndim,
                                        size_t block_size,
                                        size_t device_id) {
    auto const dev_acc = alpaka::getDevByIdx<Acc1D>(device_id);

    /* initialise<Platform>(); */

    // Create the queue
    Queue queue_(dev_acc);

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1):
        return run1(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (2):
        return run2(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (3):
        return run3(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (4):
        return run4(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (5):
        return run5(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (6):
        return run6(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (7):
        return run7(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (8):
        return run8(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (9):
        return run9(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (10):
        return run10(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
        return {};
    }
  }

  std::vector<std::vector<int>> mainRun(float dc,
                                        float rhoc,
                                        float outlier,
                                        int pPBin,
                                        const std::vector<std::vector<float>>& coords,
                                        const std::vector<float>& weights,
                                        const ExponentialKernel& kernel,
                                        int Ndim,
                                        size_t block_size,
                                        size_t device_id) {
    auto const dev_acc = alpaka::getDevByIdx<Acc1D>(device_id);

    // Create the queue
    Queue queue_(dev_acc);

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1):
        return run1(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (2):
        return run2(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (3):
        return run3(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (4):
        return run4(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (5):
        return run5(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (6):
        return run6(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (7):
        return run7(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (8):
        return run8(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (9):
        return run9(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (10):
        return run10(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
        return {};
    }
  }

  std::vector<std::vector<int>> mainRun(float dc,
                                        float rhoc,
                                        float outlier,
                                        int pPBin,
                                        const std::vector<std::vector<float>>& coords,
                                        const std::vector<float>& weights,
                                        const GaussianKernel& kernel,
                                        int Ndim,
                                        size_t block_size,
                                        size_t device_id) {
    auto const dev_acc = alpaka::getDevByIdx<Acc1D>(device_id);

    // Create the queue
    Queue queue_(dev_acc);

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1):
        return run1(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (2):
        return run2(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (3):
        return run3(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (4):
        return run4(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (5):
        return run5(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (6):
        return run6(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (7):
        return run7(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (8):
        return run8(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (9):
        return run9(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (10):
        return run10(
            dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
        return {};
    }
  }

  PYBIND11_MODULE(CLUE_GPU_CUDA, m) {
    m.doc() = "Binding of the CLUE algorithm running on CUDA GPUs";

    m.def("listDevices", &listDevices, "List the available devices for the CUDA backend");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  int,
                                  const std::vector<std::vector<float>>&,
                                  const std::vector<float>&,
                                  const FlatKernel&,
                                  int,
                                  size_t,
                                  size_t>(&mainRun),
          "mainRun");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  int,
                                  const std::vector<std::vector<float>>&,
                                  const std::vector<float>&,
                                  const ExponentialKernel&,
                                  int,
                                  size_t,
                                  size_t>(&mainRun),
          "mainRun");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  int,
                                  const std::vector<std::vector<float>>&,
                                  const std::vector<float>&,
                                  const GaussianKernel&,
                                  int,
                                  size_t,
                                  size_t>(&mainRun),
          "mainRun");
  }
};  // namespace alpaka_cuda_async
