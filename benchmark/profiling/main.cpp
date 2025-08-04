

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"

void run(const std::string& input_file) {
  const auto dev_acc = alpaka::getDevByIdx(clue::Platform{}, 0u);
  clue::Queue queue(dev_acc);

  auto h_points = clue::read_csv<2>(queue, input_file);
  clue::PointsDevice<2, clue::Device> d_points(queue, h_points.size());

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, clue::FlatKernel{.5f}, queue, block_size);
  auto clusters = algo.getClusters(h_points);
}

int main(int, char* argv[]) {
  auto input_file{std::string(argv[1])};
  run(input_file);
}
