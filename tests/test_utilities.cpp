
#include "CLUEstering/CLUEstering.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test clue::get_device utility") {
  clue::Device device = clue::get_device(0u);
  static_assert(std::is_same_v<decltype(device), clue::Device>, "Expected type clue::Device");

  const auto devices = alpaka::getDevs(clue::Platform{});
  CHECK(devices[0u] == device);
}

TEST_CASE("Test clue::get_queue utility") {
  SUBCASE("Create queue using device id") {
    const auto dev_id1 = 0;
    auto queue1 = clue::get_queue(dev_id1);
    static_assert(std::is_same_v<decltype(queue1), clue::Queue>, "Expected type clue::Queue");
    CHECK(alpaka::getDev(queue1) == alpaka::getDevByIdx(clue::Platform{}, dev_id1));

    // test both signed and unsigned integer types
    const auto dev_id2 = 0u;
    auto queue2 = clue::get_queue(dev_id2);
    static_assert(std::is_same_v<decltype(queue2), clue::Queue>, "Expected type clue::Queue");
    CHECK(alpaka::getDev(queue2) == alpaka::getDevByIdx(clue::Platform{}, dev_id2));
    CHECK(alpaka::getDev(queue2) == alpaka::getDev(queue1));

    // check if data allocation works
    clue::PointsHost<2> points1(queue1, 1000);
    auto d_points1 = clue::PointsDevice<2>(queue1, points1.size());
    clue::PointsHost<2> points2(queue2, 1000);
    auto d_points2 = clue::PointsDevice<2>(queue2, points2.size());
    CHECK(1);
  }

  SUBCASE("Create queue using device object") {
    auto device = clue::get_device(0u);

    auto queue1 = clue::get_queue(device);
    static_assert(std::is_same_v<decltype(queue1), clue::Queue>, "Expected type clue::Queue");
    CHECK(alpaka::getDev(queue1) == alpaka::getDevByIdx(clue::Platform{}, 0u));

    auto queue2 = clue::get_queue(device);
    static_assert(std::is_same_v<decltype(queue2), clue::Queue>, "Expected type clue::Queue");
    CHECK(alpaka::getDev(queue2) == alpaka::getDevByIdx(clue::Platform{}, 0u));
    CHECK(alpaka::getDev(queue2) == alpaka::getDev(queue1));

    // check if data allocation works
    clue::PointsHost<2> points1(queue1, 1000);
    auto d_points1 = clue::PointsDevice<2>(queue1, points1.size());
    clue::PointsHost<2> points2(queue2, 1000);
    auto d_points2 = clue::PointsDevice<2>(queue2, points2.size());
    CHECK(1);
  }
}

TEST_CASE("Test validation utilities") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/data_32768.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);
  auto clusters = clue::compute_clusters_points(h_points.clusterIndexes());
}
