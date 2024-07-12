
#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <ostream>
#include <vector>

#include "CLUE/CLUEAlgoAlpaka.h"
#include "CLUE/Run.h"
#include "DataFormats/Points.h"
#include "DataFormats/alpaka/PointsAlpaka.h"

#include "read_csv.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void run(const std::string& filename, float dc, float rhoc, float odf) {
    const auto dev_acc = alpaka::getDevByIdx<Acc1D>(0u);
    //const auto dev_acc = cms::alpakatools::devices<alpaka::Pltf<Device>>()[0];
    //const auto dev_acc = alpaka::getDevs<Platform>()[0];
    Queue queue_(dev_acc);
    const auto data{read_csv<float, 2>(filename)};
    Points<2> h_points(data.first, data.second);
    PointsAlpaka<2> d_points(queue_, data.second.size());

    const int pPBin{128};
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, odf, pPBin, queue_);

    const std::size_t block_size{256};
	auto result = algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue_, block_size);
  }
};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main(int argc, char* argv[]) {
  int nruns{1};
  if (argc >= 2) {
	nruns = std::stoi(argv[1]);
  }

  const std::string filename(argv[2]);
  const float dc{std::stof(argv[3])}, rhoc{std::stof(argv[4])}, outlier{std::stof(argv[5])};
  std::cout << "setup,HtD,CLD,CNH,FC,AC,DtH\n";
  for (int i{}; i < nruns; ++i) {
	ALPAKA_ACCELERATOR_NAMESPACE::run(filename, dc, rhoc, outlier);
  }
}
