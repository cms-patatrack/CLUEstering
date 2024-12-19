
#include <alpaka/alpaka.hpp>
#include <vector>

#include "../CLUE/Run.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace alpaka_tbb_async {
  void listDevices(const std::string& backend) {
    const char tab = '\t';
    const std::vector<Device> devices = alpaka::getDevs(alpaka::Platform<Acc1D>());
    if (devices.empty()) {
      std::cout << "No devices found for the " << backend << " backend." << std::endl;
      return;
    } else {
      std::cout << backend << " devices found: \n";
      for (size_t i{}; i < devices.size(); ++i) {
        std::cout << tab << "Device " << i << ": " << alpaka::getName(devices[i]) << '\n';
      }
    }
  }

  template <typename Kernel>
  std::vector<std::vector<int>> mainRun(float dc,
                                        float rhoc,
                                        float dm,
                                        int pPBin,
                                        const std::vector<std::vector<float>>& coords,
                                        const std::vector<float>& weights,
                                        const Kernel& kernel,
                                        int Ndim,
                                        size_t block_size,
                                        size_t device_id) {
    const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, device_id);

    // Create the queue
    Queue queue_(dev_acc);

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1):
        return run<1, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (2):
        return run<2, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[likely]] case (3):
        return run<3, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (4):
        return run<4, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (5):
        return run<5, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (6):
        return run<6, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (7):
        return run<7, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (8):
        return run<8, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (9):
        return run<9, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] case (10):
        return run<10, Kernel>(
            dc, rhoc, dm, pPBin, coords, weights, kernel, queue_, block_size);
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
        return {};
    }
  }

  PYBIND11_MODULE(CLUE_CPU_TBB, m) {
    m.doc() = "Binding of the CLUE algorithm running on CPU with TBB";

    m.def("listDevices", &listDevices, "List the available devices for the TBB backend");
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
                                  size_t>(&mainRun<FlatKernel>),
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
                                  size_t>(&mainRun<ExponentialKernel>),
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
                                  size_t>(&mainRun<GaussianKernel>),
          "mainRun");
  }
};  // namespace alpaka_tbb_async
