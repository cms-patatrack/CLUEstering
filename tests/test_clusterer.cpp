
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test make_cluster interfaces") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/data_32768.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  auto truth = clue::read_output<2>(queue, "../../../data/truth_files/data_32768_truth.csv");
  SUBCASE("Run clustering without passing device points") {
    algo.make_clusters(queue, h_points);

    CHECK(clue::silhouette(h_points) >= 0.9f);
  }

  SUBCASE("Run clustering without passing the queue and device points") {
    algo.make_clusters(h_points);

    CHECK(clue::silhouette(h_points) >= 0.9f);
  }
  SUBCASE("Run clustering from device points") {
    clue::copyToDevice(queue, d_points, h_points);
    algo.make_clusters(queue, d_points);
    clue::copyToHost(queue, h_points, d_points);
    alpaka::wait(queue);

    CHECK(clue::silhouette(h_points) >= 0.9f);
  }
}

TEST_CASE("Test Clusterer constructors with invalid parameters") {
  SUBCASE("Constructor without queue") {
    CHECK_THROWS(clue::Clusterer<2>(-1.f, 10.f));
    CHECK_THROWS(clue::Clusterer<2>(1.f, -10.f));
  }
  SUBCASE("Constructor with queue") {
    auto queue = clue::get_queue(0u);
    CHECK_THROWS(clue::Clusterer<2>(queue, -1.f, 10.f));
    CHECK_THROWS(clue::Clusterer<2>(queue, 1.f, -10.f));
  }
}
