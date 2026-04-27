
#include "CLUEstering/CLUEstering.hpp"

#include <algorithm>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("MahalanobisMetric: sigma controls cluster merging") {
  auto queue = clue::get_queue(0u);

  const float density_radius = 1.f;
  const float min_density = 0.3f;
  clue::Clusterer<1> algo(queue, density_radius, min_density);

  const std::vector<float> coords = {2.f, 4.f};
  const std::vector<float> weights = {1.f, 1.f};

  SUBCASE("Without sigma: Euclidean distance > seeding_distance => 2 clusters") {
    clue::PointsHost<1> points(queue, 2);
    std::ranges::copy(coords, points.coords(0).begin());
    std::ranges::copy(weights, points.weights().begin());

    algo.make_clusters(queue, points);

    CHECK(points.n_clusters() == 2);
  }

  SUBCASE("With sigma=3: Mahalanobis distance < seeding_distance => 1 cluster") {
    clue::PointsHost<1> points(queue, 2);
    std::ranges::copy(coords, points.coords(0).begin());
    std::ranges::copy(weights, points.weights().begin());

    std::vector<float> sigma = {2.f, 2.f};
    points.set_sigma(0, std::span(sigma));

    algo.make_clusters(queue, points, clue::metrics::Mahalanobis<1>{});

    CHECK(points.n_clusters() == 1);
  }
}
