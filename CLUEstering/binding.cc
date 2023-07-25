#include <vector>

#include "include/Clustering.h"
#include "include/Kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

/////////////////////
//////  Run.h  //////
/////////////////////
std::vector<std::vector<int>> run1(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<1> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run2(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<2> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run3(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<3> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run4(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<4> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run5(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<5> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run6(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<6> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run7(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<7> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run8(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<8> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run9(float dc,
                                   float rhoc,
                                   float outlier,
                                   int pPBin,
                                   std::vector<domain_t> domains,
                                   kernel const &ker,
                                   std::vector<std::vector<float>> const &coordinates,
                                   std::vector<float> const &weight) {
  ClusteringAlgo<9> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

std::vector<std::vector<int>> run10(float dc,
                                    float rhoc,
                                    float outlier,
                                    int pPBin,
                                    std::vector<domain_t> domains,
                                    kernel const &ker,
                                    std::vector<std::vector<float>> const &coordinates,
                                    std::vector<float> const &weight) {
  ClusteringAlgo<10> algo(dc, rhoc, outlier, pPBin, domains);
  algo.setPoints(coordinates[0].size(), coordinates, weight);

  return algo.makeClusters(ker);
}

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

//////////////////////////////
//////  Binding module  //////
//////////////////////////////
PYBIND11_MODULE(CLUEsteringCPP, m) {
  m.doc() = "Binding for CLUE";

  pybind11::class_<domain_t>(m, "domain_t")
	  .def(pybind11::init<>())
	  .def(pybind11::init<float,float>());

  pybind11::class_<kernel>(m, "kernel")
	  .def(pybind11::init<>())
	  .def("operator()", &kernel::operator());
  pybind11::class_<flatKernel, kernel>(m, "flatKernel")
      .def(pybind11::init<float>())
      .def("operator()", &flatKernel::operator());
  pybind11::class_<gaussianKernel, kernel>(m, "gaussianKernel")
      .def(pybind11::init<float, float, float>())
      .def("operator()", &gaussianKernel::operator());
  pybind11::class_<exponentialKernel, kernel>(m, "exponentialKernel")
      .def(pybind11::init<float, float>())
      .def("operator()", &exponentialKernel::operator());
  pybind11::class_<customKernel, kernel>(m, "customKernel")
      .def(pybind11::init<kernel_t>())
      .def("operator()", &customKernel::operator());

  m.def("mainRun", &mainRun, "mainRun");
}
