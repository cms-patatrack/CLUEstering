
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <ranges>

#include <fmt/core.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test clustering on benchmarking datasets") {
#ifdef COVERAGE
  auto range = std::make_pair(10, 11);
#else
  auto range = std::make_pair(10, 18);
#endif

  for (auto i = range.first; i < range.second; ++i) {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    clue::PointsHost<2> h_points =
        clue::read_csv<2>(queue, fmt::format("../../../data/data_{}.csv", std::pow(2, i)));
    const auto n_points = h_points.size();
    clue::PointsDevice<2> d_points(queue, n_points);

    const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.9f);
  }
}

TEST_CASE("Test clustering on aniso dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/aniso_1000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{25.f}, rhoc{5.f}, outlier{23.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);
  // TODO: use a better metric for anisotropic data
  // like Davies-Bouldin index
  // CHECK(clue::silhouette(h_points) >= 0.9f);
}

TEST_CASE("Test clustering on sissa 1000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/sissa_1000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  CHECK(clue::silhouette(h_points) >= 0.5f);
}

TEST_CASE("Test clustering on sissa 4000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/sissa_4000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  CHECK(clue::silhouette(h_points) >= 0.45f);
}

TEST_CASE("Test clustering on toy detector 1000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/toyDetector_1000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{4.f}, rhoc{2.5f}, outlier{4.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  CHECK(clue::silhouette(h_points) >= 0.8f);
}

TEST_CASE("Test clustering on blob dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<3> h_points = clue::read_csv<3>(queue, "../../../data/blob.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<3> d_points(queue, n_points);

  const float dc{1.f}, rhoc{5.f}, outlier{2.f};
  clue::Clusterer<3> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  CHECK(clue::silhouette(h_points) >= 0.8f);
}

TEST_CASE("Test clustering on data with periodic coordinates") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> points = clue::read_csv<2>(queue, "../../../data/opposite_angles.csv");
  const float dc{.2f}, rhoc{5.f}, outlier{.2f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.setWrappedCoordinates(0, 1);
  algo.make_clusters(queue, points);
  // TODO: reimplement wrapped coordinates before 2.9.0
  // CHECK(points.n_clusters() == 1);
}
