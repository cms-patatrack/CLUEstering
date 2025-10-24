
#include "CLUEstering/CLUEstering.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test computation of cluster centroid") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/sissa.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{21.f}, rhoc{10.f}, outlier{21.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  SUBCASE("Check centroid of cluster 0") { volatile auto _ = clue::cluster_centroid(h_points, 0); }
}

TEST_CASE("Test computation of all cluster centroids") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/sissa.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{21.f}, rhoc{10.f}, outlier{21.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);
  auto centroids = clue::cluster_centroids(h_points);
  auto clusters = clue::get_clusters(h_points);
  CHECK(centroids.size() == clusters.size());
}
