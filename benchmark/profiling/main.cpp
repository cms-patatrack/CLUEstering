

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/DataFormats/PointsHost.hpp"
#include "CLUEstering/DataFormats/PointsDevice.hpp"

#include "CLUEstering/utility/read_csv.hpp"

using ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Acc1D;
using ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Device;
using ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Queue;

void run(const std::string& input_file) {
  auto coords = read_csv<float, 2>(input_file);
  const auto n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<2> h_points(queue, n_points, coords.data(), results.data());
  clue::PointsDevice<2, Device> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = algo.getClusters(h_points);
}

int main(int, char* argv[]) {
  auto input_file{std::string(argv[1])};
  run(input_file);
}
