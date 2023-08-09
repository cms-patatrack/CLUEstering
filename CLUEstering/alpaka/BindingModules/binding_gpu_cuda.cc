
#include <vector>

#include "include/Clustering.h"
#include "include/Kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

std::vector<std::vector<int>> mainRun(float dc,
                                      float rhoc,
                                      float outlier,
                                      int pPBin,
                                      std::vector<domain_t> domains,
                                      kernel const &ker,
                                      std::vector<std::vector<float>> const &coords,
                                      std::vector<float> const &weight,
                                      int Ndim) {
  // Running the clustering algorithm //
  switch (Ndim) {
    [[unlikely]] case (1) :
      return run1(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[likely]] case (2) :
      return run2(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[likely]] case (3) :
      return run3(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] case (4) :
      return run4(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] case (5) :
      return run5(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] case (6) :
      return run6(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] case (7) :
      return run7(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] case (8) :
      return run8(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] case (9) :
      return run9(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] case (10) :
      return run10(dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
      break;
    [[unlikely]] default:
      std::cout << "This library only works up to 10 dimensions\n";
      return {};
      break;
  }
}

PYBIND11_MODULE(CLUE_CPUSerial, m) {
  m.doc() = "Binding for CLUE";

  m.def("mainRun", &mainRun, "mainRun");
}
