
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utility/read_csv.hpp"
#include "CLUEstering/utility/validation.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#include "doctest.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

TEST_CASE("Test make_cluster interfaces") {
  auto coords = read_csv<float, 2>("../data/data_32768.csv");
  const uint32_t n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<2> h_points(queue, n_points, coords, results);
  clue::PointsDevice<2, Device> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  const std::size_t block_size{256};

  const auto truth_data = read_output<2>("../data/truth_files/data_32768_truth.csv");
  auto truth_ids = std::span<const int>{truth_data.data(), n_points};
  auto truth_isSeed = std::span<const int>{truth_data.data() + n_points, n_points};
  SUBCASE("Run clustering without passing device points") {
    algo.make_clusters(h_points, FlatKernel{.5f}, queue, block_size);
    auto clusters = std::span<const int>{results.data(), n_points};
    auto isSeed = std::span<const int>(results.data() + n_points, n_points);

    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }

  SUBCASE("Run clustering without passing the queue") {
    algo.make_clusters(h_points, d_points, FlatKernel{.5f}, block_size);
    auto clusters = std::span<const int>{results.data(), n_points};
    auto isSeed = std::span<const int>(results.data() + n_points, n_points);

    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }

  SUBCASE("Run clustering without passing the queue and device points") {
    algo.make_clusters(h_points, FlatKernel{.5f}, block_size);
    auto clusters = std::span<const int>{results.data(), n_points};
    auto isSeed = std::span<const int>(results.data() + n_points, n_points);

    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }
}
