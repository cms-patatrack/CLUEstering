
#include "CLUEstering/CLUEstering.hpp"

#include "doctest.h"

TEST_CASE("Test validation scores on toy detector dataset") {
  auto queue = clue::get_queue(0u);
  clue::PointsHost<2> points = clue::read_csv<2>(queue, "../data/toyDetector.csv");
  clue::Clusterer<2> clusterer(queue, 4.f, 2.5f, 4.f);
  clusterer.make_clusters(queue, points);

  SUBCASE("Test computation of silhouette score on all points singularly") {
    for (auto i = 0; i < points.size(); ++i) {
      const auto silhouette = clue::silhouette(points, i);
      CHECK(silhouette >= -1.f);
      CHECK(silhouette <= 1.f);
    }
  }
  SUBCASE("Test computation of silhouette score on all dataset") {
    const auto silhouette = clue::silhouette(points);
    CHECK(silhouette >= -1.f);
    CHECK(silhouette <= 1.f);
  }
}
