
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
  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/data_32768.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2, Device> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  const std::size_t block_size{256};

  const auto truth_data = clue::read_output<2>(queue, "../data/truth_files/data_32768_truth.csv");
  auto truth_ids = truth_data.clusterIndexes();
  auto truth_isSeed = truth_data.isSeed();
  SUBCASE("Run clustering without passing device points") {
    algo.make_clusters(h_points, FlatKernel{.5f}, queue, block_size);
    auto clusters = h_points.clusterIndexes();
    auto isSeed = h_points.isSeed();

    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }

  SUBCASE("Run clustering without passing the queue") {
    algo.make_clusters(h_points, d_points, FlatKernel{.5f}, block_size);
    auto clusters = h_points.clusterIndexes();
    auto isSeed = h_points.isSeed();

    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }

  SUBCASE("Run clustering without passing the queue and device points") {
    algo.make_clusters(h_points, FlatKernel{.5f}, block_size);
    auto clusters = h_points.clusterIndexes();
    auto isSeed = h_points.isSeed();

    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }
  SUBCASE("Run clustering from device points") {
    clue::copyToDevice(queue, d_points, h_points);
    algo.make_clusters(d_points, FlatKernel{.5f}, queue, block_size);
    clue::copyToHost(queue, h_points, d_points);

    auto clusters = h_points.clusterIndexes();
    auto isSeed = h_points.isSeed();

    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }
}
