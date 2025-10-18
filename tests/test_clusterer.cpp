
#include "CLUEstering/CLUEstering.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#include "doctest.h"

TEST_CASE("Test make_cluster interfaces") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/data_32768.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  const std::size_t block_size{256};

  const auto truth_data = clue::read_output<2>(queue, "../data/truth_files/data_32768_truth.csv");
  auto truth_ids = truth_data.clusterIndexes();
  SUBCASE("Run clustering without passing device points") {
    algo.make_clusters(queue, h_points, clue::FlatKernel{.5f}, block_size);
    auto clusters = h_points.clusterIndexes();

    CHECK(clue::validate_results(clusters, truth_ids));
  }

  SUBCASE("Run clustering without passing the queue") {
    algo.make_clusters(h_points, d_points, clue::FlatKernel{.5f}, block_size);
    auto clusters = h_points.clusterIndexes();

    CHECK(clue::validate_results(clusters, truth_ids));
  }

  SUBCASE("Run clustering without passing the queue and device points") {
    algo.make_clusters(h_points, clue::FlatKernel{.5f}, block_size);
    auto clusters = h_points.clusterIndexes();

    CHECK(clue::validate_results(clusters, truth_ids));
  }
  SUBCASE("Run clustering from device points") {
    clue::copyToDevice(queue, d_points, h_points);
    algo.make_clusters(queue, d_points, clue::FlatKernel{.5f}, block_size);
    clue::copyToHost(queue, h_points, d_points);

    auto clusters = h_points.clusterIndexes();

    CHECK(clue::validate_results(clusters, truth_ids));
  }
}

TEST_CASE("Test Clusterer constructors with invalid parameters") {
  SUBCASE("Constructor without queue") {
    CHECK_THROWS(clue::Clusterer<2>(-1.f, 10.f, 1.5f));
    CHECK_THROWS(clue::Clusterer<2>(1.f, -10.f, 1.5f));
    CHECK_THROWS(clue::Clusterer<2>(1.f, 10.f, -1.5f));
  }
  SUBCASE("Constructor with queue") {
    auto queue = clue::get_queue(0u);
    CHECK_THROWS(clue::Clusterer<2>(queue, -1.f, 10.f, 1.5f));
    CHECK_THROWS(clue::Clusterer<2>(queue, 1.f, -10.f, 1.5f));
    CHECK_THROWS(clue::Clusterer<2>(queue, 1.f, 10.f, -1.5f));
  }
}
