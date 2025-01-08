
#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>

#include "CLUE/CLUEAlgoAlpaka.h"
#include "CLUE/Run.h"
#include "DataFormats/Points.h"
#include "DataFormats/alpaka/PointsAlpaka.h"

#include "read_csv.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {
  void run(const std::string& input_file) {
    const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
    Queue queue_(dev_acc);
    const auto data{read_csv<float, 2>(input_file)};
    Points<2> h_points(data.first, data.second);
    PointsAlpaka<2> d_points(queue_, data.second.size());

    const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
    const int pPBin{128};
    CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue_);

    const std::size_t block_size{256};
    auto result =
        algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue_, block_size);
  }
};  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE

int main(int argc, char* argv[]) {
  auto input_file{std::string(argv[1])};
  using ALPAKA_ACCELERATOR_NAMESPACE_CLUE;
  run(input_file);
}
