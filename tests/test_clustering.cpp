
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

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
        clue::read_csv<2>(queue, fmt::format("../data/data_{}.csv", std::pow(2, i)));
    const auto n_points = h_points.size();
    clue::PointsDevice<2> d_points(queue, n_points);

    const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    auto truth = clue::read_output<2>(
        queue, fmt::format("../data/truth_files/data_{}_truth.csv", std::pow(2, i)));
    CHECK(clue::validate_results(h_points, truth));
  }
}

TEST_CASE("Test clustering on aniso dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/aniso_1000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{25.f}, rhoc{5.f}, outlier{23.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  auto truth = clue::read_output<2>(queue, "../data/truth_files/aniso_1000_truth.csv");
  CHECK(clue::validate_results(h_points, truth));
}

TEST_CASE("Test clustering on sissa 1000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/sissa_1000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{21.f}, rhoc{10.f}, outlier{21.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  auto truth = clue::read_output<2>(queue, "../data/truth_files/sissa_1000_truth.csv");
  CHECK(clue::validate_results(h_points, truth));
}

TEST_CASE("Test clustering on sissa 4000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/sissa_4000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  auto truth = clue::read_output<2>(queue, "../data/truth_files/sissa_4000_truth.csv");
  CHECK(clue::validate_results(h_points, truth));
}

TEST_CASE("Test clustering on toy detector 1000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/toyDetector_1000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{4.f}, rhoc{2.5f}, outlier{4.f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  auto truth = clue::read_output<2>(queue, "../data/truth_files/toy_det_1000_truth.csv");
  CHECK(clue::validate_results(h_points, truth));
}

TEST_CASE("Test clustering on toy detector 5000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/toyDetector_5000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{2.5f}, rhoc{2.f}, outlier{7.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  auto truth = clue::read_output<2>(queue, "../data/truth_files/toy_det_5000_truth.csv");
  CHECK(clue::validate_results(h_points, truth));
}

TEST_CASE("Test clustering on toy detector 10000 dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/toyDetector_10000.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{2.5f}, rhoc{2.f}, outlier{7.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  auto truth = clue::read_output<2>(queue, "../data/truth_files/toy_det_10000_truth.csv");
  CHECK(clue::validate_results(h_points, truth));
}

TEST_CASE("Test clustering on blob dataset") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<3> h_points = clue::read_csv<3>(queue, "../data/blob.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<3> d_points(queue, n_points);

  const float dc{1.f}, rhoc{5.f}, outlier{2.f};
  clue::Clusterer<3> algo(queue, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);

  auto truth = clue::read_output<3>(queue, "../data/truth_files/blobs_truth.csv");
  CHECK(clue::validate_results(h_points, truth));
}

TEST_CASE("Test clustering on data with periodic coordinates") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> points = clue::read_csv<2>(queue, "../data/opposite_angles.csv");
  const float dc{.2f}, rhoc{5.f}, outlier{.2f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  algo.setWrappedCoordinates(0, 1);
  algo.make_clusters(queue, points);
  CHECK(points.n_clusters() == 1);
}
