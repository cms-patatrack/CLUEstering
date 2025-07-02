
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utility/read_csv.hpp"
#include "CLUEstering/utility/validation.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#include <fmt/core.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

TEST_CASE("Test clustering on benchmarking datasets") {
  for (auto i = 10; i < 19; ++i) {
    const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
    Queue queue(dev_acc);

    clue::PointsHost<2> h_points =
        clue::read_csv<2>(queue, fmt::format("../data/data_{}.csv", std::pow(2, i)));
    const auto n_points = h_points.size();
    clue::PointsDevice<2, Device> d_points(queue, n_points);

    const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    const std::size_t block_size{256};
    algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
    auto clusters = h_points.clusterIndexes();
    auto isSeed = h_points.isSeed();

    const auto truth_data = clue::read_output<2>(
        queue, fmt::format("../data/truth_files/data_{}_truth.csv", std::pow(2, i)));
    auto truth_ids = truth_data.clusterIndexes();
    auto truth_isSeed = truth_data.isSeed();
    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }
}

TEST_CASE("Test clustering on sissa") {
  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/sissa.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2, Device> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = h_points.clusterIndexes();
  auto isSeed = h_points.isSeed();

  const auto truth_data = clue::read_output<2>(queue, "../data/truth_files/sissa_1000_truth.csv");
  auto truth_ids = truth_data.clusterIndexes();
  auto truth_isSeed = truth_data.isSeed();
  CHECK(clue::validate_results(clusters, truth_ids));
  CHECK(std::ranges::equal(truth_isSeed, isSeed));
}

TEST_CASE("Test clustering on toy detector dataset") {
  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/toyDetector.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2, Device> d_points(queue, n_points);

  const float dc{4.5f}, rhoc{2.5f}, outlier{4.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = h_points.clusterIndexes();
  auto isSeed = h_points.isSeed();

  const auto truth_data = clue::read_output<2>(queue, "../data/truth_files/toy_det_1000_truth.csv");
  auto truth_ids = truth_data.clusterIndexes();
  auto truth_isSeed = truth_data.isSeed();
  CHECK(clue::validate_results(clusters, truth_ids));
  CHECK(std::ranges::equal(truth_isSeed, isSeed));
}

TEST_CASE("Test clustering on blob dataset") {
  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<3> h_points = clue::read_csv<3>(queue, "../data/blob.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<3, Device> d_points(queue, n_points);

  const float dc{1.f}, rhoc{5.f}, outlier{2.f};
  clue::Clusterer<3> algo(queue, dc, rhoc, outlier);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = h_points.clusterIndexes();
  auto isSeed = h_points.isSeed();

  const auto truth_data = clue::read_output<3>(queue, "../data/truth_files/blobs_truth.csv");
  auto truth_ids = truth_data.clusterIndexes();
  auto truth_isSeed = truth_data.isSeed();
  CHECK(clue::validate_results(clusters, truth_ids));
  CHECK(std::ranges::equal(truth_isSeed, isSeed));
}
