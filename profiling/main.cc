
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
  void run() {
    const auto dev_acc = alpaka::getDevByIdx<Acc1D>(0u);
    Queue queue_(dev_acc);

    const auto data{read_csv<float, 3>("./blob.csv")};
    Points<3> h_points(data.first, data.second);
    PointsAlpaka<3> d_points(queue_, data.second.size());

    const float dc{3.f}, rhoc{2.f}, outlier{3.5f};
    const int pPBin{10};
    CLUEAlgoAlpaka<Acc1D, 3> algo(dc, rhoc, outlier, pPBin, queue_);

    const std::size_t block_size{256};
    auto result = algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue_, block_size);
    std::cout << "Number of clusters: " << std::accumulate(result[1].begin(), result[1].end(), 0)
              << std::endl;
  }
};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  ALPAKA_ACCELERATOR_NAMESPACE::run();
}
