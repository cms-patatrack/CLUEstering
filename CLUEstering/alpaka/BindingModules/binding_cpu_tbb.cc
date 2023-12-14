
#include <alpaka/alpaka.hpp>
#include <vector>

#include "../CLUE/CLUEAlgoAlpaka.h"
#include "../CLUE/Run.h"
#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

namespace alpaka_tbb_async {
  std::vector<std::vector<int>> mainRun(float dc,
                                        float rhoc,
                                        float outlier,
                                        int pPBin,
                                        const std::vector<std::vector<float>>& coords,
                                        const std::vector<float>& weights,
										const FlatKernel& kernel,
                                        int Ndim,
										size_t block_size) {
    auto const dev_acc = alpaka::getDevByIdx<Acc1D>(0u);

    // Create the queue
    Queue queue_(dev_acc);

    /* Vec const elementsPerThread(Vec::all(static_cast<Idx>(1))); */
    /* Vec const threadsPerGrid(Vec::all(static_cast<Idx>(8))); */
    /* WorkDiv const work_div = */
    /*     alpaka::getValidWorkDiv<Acc1D>(dev_acc, */
    /*                                  threadsPerGrid, */
    /*                                  elementsPerThread, */
    /*                                  false, */
    /*                                  alpaka::GridBlockExtentSubDivRestrictions::Unrestricted); */

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1) :
        /* return run1(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[likely]] case (2) :
        return run2(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
        break;
      [[likely]] case (3) :
        /* return run3(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (4) :
        /* return run4(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (5) :
        /* return run5(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (6) :
        /* return run6(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (7) :
        /* return run7(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (8) :
        /* return run8(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (9) :
        /* return run9(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (10) :
        /* return run10(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
        return {};
        break;
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
										size_t block_size) {
    auto const dev_acc = alpaka::getDevByIdx<Acc1D>(0u);

    // Create the queue
    Queue queue_(dev_acc);

    /* Vec const elementsPerThread(Vec::all(static_cast<Idx>(1))); */
    /* Vec const threadsPerGrid(Vec::all(static_cast<Idx>(8))); */
    /* WorkDiv const work_div = */
    /*     alpaka::getValidWorkDiv<Acc1D>(dev_acc, */
    /*                                  threadsPerGrid, */
    /*                                  elementsPerThread, */
    /*                                  false, */
    /*                                  alpaka::GridBlockExtentSubDivRestrictions::Unrestricted); */

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1) :
        /* return run1(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[likely]] case (2) :
        return run2(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
        break;
      [[likely]] case (3) :
        /* return run3(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (4) :
        /* return run4(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (5) :
        /* return run5(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (6) :
        /* return run6(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (7) :
        /* return run7(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (8) :
        /* return run8(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (9) :
        /* return run9(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (10) :
        /* return run10(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
        return {};
        break;
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
										size_t block_size) {
    auto const dev_acc = alpaka::getDevByIdx<Acc1D>(0u);

    // Create the queue
    Queue queue_(dev_acc);

    /* Vec const elementsPerThread(Vec::all(static_cast<Idx>(1))); */
    /* Vec const threadsPerGrid(Vec::all(static_cast<Idx>(8))); */
    /* WorkDiv const work_div = */
    /*     alpaka::getValidWorkDiv<Acc1D>(dev_acc, */
    /*                                  threadsPerGrid, */
    /*                                  elementsPerThread, */
    /*                                  false, */
    /*                                  alpaka::GridBlockExtentSubDivRestrictions::Unrestricted); */

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1) :
        /* return run1(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[likely]] case (2) :
        return run2(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_, block_size);
        break;
      [[likely]] case (3) :
        /* return run3(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (4) :
        /* return run4(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (5) :
        /* return run5(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (6) :
        /* return run6(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (7) :
        /* return run7(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (8) :
        /* return run8(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (9) :
        /* return run9(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] case (10) :
        /* return run10(dc, rhoc, outlier, pPBin, coords, weights, kernel, queue_); */
        break;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
        return {};
        break;
    }
  }

  PYBIND11_MODULE(CLUE_CPU_TBB, m) {
    m.doc() = "Binding of the CLUE algorithm running on CPU with TBB";

    /* pybind11::class_<ConvolutionalKernel>(m, "ConvolutionalKernel").def(pybind11::init<>()) */
		/* .def("operator()", &ConvolutionalKernel::operator()); */
    /* pybind11::class_<FlatKernel, ConvolutionalKernel>(m, "FlatKernel") */
    /*     .def(pybind11::init<float>()) */
    /*     .def("operator()", &FlatKernel::operator()); */
    /* pybind11::class_<GaussianKernel, ConvolutionalKernel>(m, "GaussianKernel") */
    /*     .def(pybind11::init<float, float, float>()) */
    /*     .def("operator()", &GaussianKernel::operator()); */
    /* pybind11::class_<ExponentialKernel, ConvolutionalKernel>(m, "ExponentialKernel") */
    /*     .def(pybind11::init<float, float>()) */
    /*     .def("operator()", &ExponentialKernel::operator()); */
    /* pybind11::class_<CustomKernel, ConvolutionalKernel>(m, "CustomKernel") */
    /*     .def(pybind11::init<kernel_t>()) */
    /*     .def("operator()", &CustomKernel::operator()); */
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  int,
                                  const std::vector<std::vector<float>>&,
                                  const std::vector<float>&,
                                  const FlatKernel&,
                                  int,
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
								  size_t>(&mainRun),
          "mainRun");

    /* m.def("mainRun", &mainRun, "mainRun"); */
  }
};  // namespace alpaka_tbb_async
