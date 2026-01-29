#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <numbers>
#include <ranges>

#include <fmt/core.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test clustering on benchmarking datasets with double precision") {
#ifdef COVERAGE
  auto range = std::make_pair(10, 14);
#else
  auto range = std::make_pair(10, 18);
#endif

  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const double dc{1.5}, rhoc{10.}, outlier{1.5};
  clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

  for (auto i = range.first; i < range.second; ++i) {
    const auto test_file_path =
        std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
    clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2, double> d_points(queue, n_points);

    algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<2, double>{}, clue::FlatKernel<double>{0.5});

    CHECK(clue::silhouette(h_points) >= 0.9);
  }
}

TEST_CASE("Test clustering on aniso dataset with double precision") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
  clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2, double> d_points(queue, n_points);

  const double dc{25.}, rhoc{5.}, outlier{23.};
  clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<2, double>{}, clue::FlatKernel<double>{0.5});
  // TODO: use a better metric for anisotropic data
  // like Davies-Bouldin index
  // CHECK(clue::silhouette(h_points) >= 0.9);
}

TEST_CASE("Test clustering on sissa 1000 dataset with double precision") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
  clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2, double> d_points(queue, n_points);

  const double dc{20.}, rhoc{10.}, outlier{20.};
  clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<2, double>{}, clue::FlatKernel<double>{0.5});

  CHECK(clue::silhouette(h_points) >= 0.5);
}

TEST_CASE("Test clustering on sissa 4000 dataset with double precision") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
  clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2, double> d_points(queue, n_points);

  const double dc{20.}, rhoc{10.}, outlier{20.};
  clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<2, double>{}, clue::FlatKernel<double>{0.5});

  CHECK(clue::silhouette(h_points) >= 0.45);
}

TEST_CASE("Test clustering on toy detector 1000 dataset with double precision") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
  clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2, double> d_points(queue, n_points);

  const double dc{4.}, rhoc{2.5}, outlier{4.};
  clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<2, double>{}, clue::FlatKernel<double>{0.5});

  CHECK(clue::silhouette(h_points) >= 0.8);
}

TEST_CASE("Test clustering on blob dataset with double precision") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
  clue::PointsHost<3, double> h_points = clue::read_csv<3, double>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<3, double> d_points(queue, n_points);

  const double dc{1.}, rhoc{5.}, outlier{2.};
  clue::Clusterer<3, double> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<3, double>{}, clue::FlatKernel<double>{0.5});

  CHECK(clue::silhouette(h_points) >= 0.8);
}

TEST_CASE("Test clustering on data with periodic coordinates and double precision") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
  clue::PointsHost<2, double> points = clue::read_csv<2, double>(queue, test_file_path);
  const double dc{.2}, rhoc{5.}, outlier{.2};
  clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

  algo.setWrappedCoordinates(0, 1);
  algo.make_clusters(
      queue,
      points,
      clue::metrics::PeriodicEuclidean<2, double>(std::array<double, 2>{0., 2. * std::numbers::pi_v<double>}),
      clue::FlatKernel<double>{0.5});
  // TODO: reimplement wrapped coordinates before 2.9.0
  CHECK(points.n_clusters() == 1);
}
