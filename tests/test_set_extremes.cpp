#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <array>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("Test clustering with user-defined extremes") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};

  // the cluster labels depend on the order of execution on parallel backends,
  // so the results are compared through the number of clusters
  clue::PointsHost<2> h_points_default = clue::read_csv<2, float>(queue, test_file_path);
  clue::Clusterer<2> algo_default(queue, dc, rhoc, outlier);
  algo_default.make_clusters(queue, h_points_default);
  const auto n_clusters_default = h_points_default.n_clusters();

  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  SUBCASE("Extremes passed as a braced list") {
    algo.setExtremes({{-100.f, 100.f}, {-100.f, 100.f}});
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == n_clusters_default);
    CHECK(clue::silhouette(h_points) >= 0.9f);
  }
  SUBCASE("Extremes passed as a std::array") {
    algo.setExtremes(std::array{std::pair{-100.f, 100.f}, std::pair{-100.f, 100.f}});
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == n_clusters_default);
  }
  SUBCASE("Extremes passed as a std::vector") {
    const std::vector<std::pair<float, float>> extremes{{-100.f, 100.f}, {-100.f, 100.f}};
    algo.setExtremes(extremes);
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == n_clusters_default);
  }
  SUBCASE("Extremes passed as a std::span") {
    const std::array<std::pair<float, float>, 2> extremes{{{-100.f, 100.f}, {-100.f, 100.f}}};
    algo.setExtremes(std::span{extremes});
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == n_clusters_default);
  }
  SUBCASE("Extremes of a different floating point type") {
    algo.setExtremes(std::array{std::pair{-100., 100.}, std::pair{-100., 100.}});
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == n_clusters_default);
  }
  SUBCASE("Extremes larger than the data") {
    algo.setExtremes({{-1000.f, 1000.f}, {-1000.f, 1000.f}});
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == n_clusters_default);
  }
  SUBCASE("Extremes smaller than the data") {
    // the points outside of the extremes are assigned to the tiles on the border,
    // which is slower but gives the same result
    algo.setExtremes({{-50.f, 50.f}, {-50.f, 50.f}});
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == n_clusters_default);
  }
  SUBCASE("Extremes are kept between consecutive runs") {
    algo.setExtremes({{-100.f, 100.f}, {-100.f, 100.f}});
    algo.make_clusters(queue, h_points);
    CHECK(h_points.n_clusters() == n_clusters_default);

    clue::PointsHost<2> h_points_second = clue::read_csv<2, float>(queue, test_file_path);
    algo.make_clusters(queue, h_points_second);
    CHECK(h_points_second.n_clusters() == n_clusters_default);
  }
  SUBCASE("Extremes set with device points") {
    clue::PointsDevice<2> d_points(queue, h_points.size());
    clue::copyToDevice(queue, d_points, h_points);
    algo.setExtremes({{-100.f, 100.f}, {-100.f, 100.f}});
    algo.make_clusters(queue, d_points);
    clue::copyToHost(queue, h_points, d_points);
    alpaka::wait(queue);

    CHECK(h_points.n_clusters() == n_clusters_default);
  }
}

TEST_CASE("Test clustering with user-defined extremes in double precision") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  const auto dc{1.3}, rhoc{10.}, outlier{1.3};

  clue::PointsHost<2, double> h_points_default = clue::read_csv<2, double>(queue, test_file_path);
  clue::Clusterer<2, double> algo_default(queue, dc, rhoc, outlier);
  algo_default.make_clusters(queue, h_points_default);

  clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
  clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);
  algo.setExtremes({{-100., 100.}, {-100., 100.}});
  algo.make_clusters(queue, h_points);

  CHECK(h_points.n_clusters() == h_points_default.n_clusters());
}

TEST_CASE("Test batched clustering with user-defined extremes") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points =
      clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/batched_data_1024.csv");
  clue::PointsDevice<2> d_points(queue, h_points.size());

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  algo.setExtremes({{-100.f, 100.f}, {-100.f, 100.f}});

  std::vector<uint32_t> event_sizes(10, 1024);
  algo.make_clusters(queue, h_points, d_points, event_sizes);

  auto truth = clue::read_output<2, float>(
      queue, std::string(TEST_DATA_DIR) + "/truth_files/data_1024_truth.csv");
  CHECK(h_points.n_clusters() == truth.n_clusters() * 10);
}

TEST_CASE("Test clustering with user-defined extremes on periodic coordinates") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
  clue::PointsHost<2> points = clue::read_csv<2, float>(queue, test_file_path);
  const float dc{.2f}, rhoc{5.f}, outlier{.2f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  // the extremes of the periodic coordinate define its exact period
  constexpr auto pi = std::numbers::pi_v<float>;
  algo.setWrappedCoordinates(0, 1);
  algo.setExtremes({{0.f, 2.f}, {-pi, pi}});
  algo.make_clusters(
      queue, points, clue::metrics::PeriodicEuclidean(std::array<float, 2>{0.f, 2.f * pi}));

  CHECK(points.n_clusters() == 1);
}

TEST_CASE("Test setExtremes with invalid extremes") {
  clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);

  SUBCASE("Minimum larger than the maximum") {
    CHECK_THROWS(algo.setExtremes({{100.f, -100.f}, {-100.f, 100.f}}));
  }
  SUBCASE("Minimum equal to the maximum") {
    CHECK_THROWS(algo.setExtremes({{-100.f, 100.f}, {1.f, 1.f}}));
  }
  SUBCASE("Missing extremes for one of the dimensions") {
    CHECK_THROWS(algo.setExtremes({{-100.f, 100.f}}));
  }
  SUBCASE("Range with the wrong number of dimensions") {
    const std::vector<std::pair<float, float>> extremes{
        {-100.f, 100.f}, {-100.f, 100.f}, {-100.f, 100.f}};
    CHECK_THROWS(algo.setExtremes(extremes));
  }
}
