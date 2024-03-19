
#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>

#include "CLUE/CLUEAlgoAlpaka.h"
#include "CLUE/Run.h"
#include "DataFormats/Points.h"
#include "DataFormats/alpaka/PointsAlpaka.h"

#include "read_csv.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void run(int nruns) {
#ifdef ANNOTATE
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();

    start = std::chrono::high_resolution_clock::now();
#endif
    const auto dev_acc = alpaka::getDevByIdx<Acc1D>(0u);
    //const auto dev_acc = cms::alpakatools::devices<alpaka::Pltf<Device>>()[0];
    //const auto dev_acc = alpaka::getDevs<Platform>()[0];
#ifdef ANNOTATE
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "getdev:\n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
#endif
    Queue queue_(dev_acc);
#ifdef ANNOTATE
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "queue:\n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
#endif
    const auto data{read_csv<float, 3>("./blob.csv")};
#ifdef ANNOTATE
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "read csv:\n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
#endif
    Points<3> h_points(data.first, data.second);
#ifdef ANNOTATE
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "host points:\n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
#endif
    PointsAlpaka<3> d_points(queue_, data.second.size());
#ifdef ANNOTATE
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "alpaka points:\n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << std::endl;
#endif

    const float dc{3.f}, rhoc{2.f}, outlier{3.5f};
    const int pPBin{10};
#ifdef ANNOTATE
    start = std::chrono::high_resolution_clock::now();
#endif
    CLUEAlgoAlpaka<Acc1D, 3> algo(dc, rhoc, outlier, pPBin, queue_);
#ifdef ANNOTATE
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "create algo:\n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << std::endl;
#endif

    const std::size_t block_size{256};
#ifdef ANNOTATE
    start = std::chrono::high_resolution_clock::now();
#endif
    for (int i{}; i < nruns; ++i) {
      auto result = algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue_, block_size);
    }
    //std::cout << "Number of clusters: " << std::accumulate(result[1].begin(), result[1].end(), 0)
    //          << std::endl;
#ifdef ANNOTATE
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "make clusters 10 times:\n";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << std::endl;
#endif
  }
};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main(int argc, char* argv[]) {
  int nruns{1};
  if (argc == 2) {
	nruns = std::stoi(argv[1]);

  }
  ALPAKA_ACCELERATOR_NAMESPACE::run(nruns);
}
