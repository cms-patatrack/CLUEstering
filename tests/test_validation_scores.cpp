
#include "CLUEstering/CLUEstering.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test validation scores on toy detector dataset") {
  auto queue = clue::get_queue(0u);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
  clue::PointsHost<2> points = clue::read_csv<2>(queue, test_file_path);
  clue::Clusterer<2> clusterer(queue, 4.f, 2.5f, 4.f);
  clusterer.make_clusters(queue, points);

  SUBCASE("Test computation of silhouette score on all points singularly") {
    for (auto i = 0; i < points.size(); ++i) {
      if (points[i].cluster_index() < 0)  // Skip noise points
        continue;
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

  SUBCASE("Test computation of davies-bouldin score") {
    const auto davies_bouldin = clue::davies_bouldin(points);
	std::cout << "Davies-Bouldin score: " << davies_bouldin << std::endl;
    CHECK(davies_bouldin >= 0.f);
  }
}
