
#include "CLUEstering/CLUEstering.hpp"

#include <cmath>
#include <fstream>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include "cuda/cuda_check.hpp"
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include "hip/hip_check.hpp"
#endif

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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  SUBCASE("Create queue using CUDA stream") {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    {
      auto queue = clue::get_queue(stream);
      static_assert(std::is_same_v<decltype(queue), clue::Queue>, "Expected type clue::Queue");
      CHECK(alpaka::getDev(queue) == alpaka::getDevByIdx(clue::Platform{}, 0u));

      // check if data allocation works
      clue::PointsHost<2> points1(queue, 1000);
      auto d_points1 = clue::PointsDevice<2>(queue, points1.size());
      CHECK(1);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  SUBCASE("Create queue using HIP stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    {
      auto queue = clue::get_queue(stream);
      static_assert(std::is_same_v<decltype(queue), clue::Queue>, "Expected type clue::Queue");
      CHECK(alpaka::getDev(queue) == alpaka::getDevByIdx(clue::Platform{}, 0u));

      // check if data allocation works
      clue::PointsHost<2> points1(queue, 1000);
      auto d_points1 = clue::PointsDevice<2>(queue, points1.size());
      CHECK(1);
    }

    HIP_CHECK(hipStreamDestroy(stream));
  }
#endif
}

TEST_CASE("Test get_clusters host function") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);
  auto clusters = clue::get_clusters(h_points);
}

TEST_CASE("Test get_clusters device function") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);
  auto clusters = clue::get_clusters(queue, d_points);
}

TEST_CASE("Test I/O with non-existent file") {
  SUBCASE("read_csv with non-existent file") {
    auto queue = clue::get_queue(0u);

    const std::string invalid_file_path = "non_existent_file.csv";
    auto invalid_call = [&] { clue::read_csv<2, float>(queue, invalid_file_path); };

    CHECK_THROWS_AS(invalid_call(), std::runtime_error);
  }
  SUBCASE("read_output with non-existent file") {
    auto queue = clue::get_queue(0u);

    const std::string invalid_file_path = "non_existent_file.csv";
    auto invalid_call = [&] { clue::read_output<2, float>(queue, invalid_file_path); };

    CHECK_THROWS_AS(invalid_call(), std::runtime_error);
  }
}

namespace {

  // Writes the content to a file in the current directory and returns its path
  std::string write_file(const std::string& name, const std::string& content) {
    std::ofstream file(name, std::ios::binary);
    file << content;
    return name;
  }

}  // namespace

TEST_CASE("Test read_csv with different file endings") {
  auto queue = clue::get_queue(0u);

  const std::string header = "x0,x1,weight\n";
  const std::string rows = "1.,2.,3.\n4.,5.,6.\n7.,8.,9.";
  const auto check = [](const clue::PointsHost<2>& points) {
    REQUIRE(points.size() == 3);
    CHECK(points.coords(0)[2] == 7.f);
    CHECK(points.coords(1)[2] == 8.f);
    CHECK(points.weights()[2] == 9.f);
  };

  SUBCASE("Trailing newline") {
    check(clue::read_csv<2, float>(queue, write_file("newline.csv", header + rows + "\n")));
  }
  SUBCASE("No trailing newline") {
    check(clue::read_csv<2, float>(queue, write_file("no_newline.csv", header + rows)));
  }
  SUBCASE("Blank lines at the end") {
    check(
        clue::read_csv<2, float>(queue, write_file("blank_lines.csv", header + rows + "\n\n \n")));
  }
  SUBCASE("Windows line endings") {
    std::string content = header + rows + "\n";
    for (auto pos = content.find('\n'); pos != std::string::npos;
         pos = content.find('\n', pos + 2)) {
      content.insert(pos, "\r");
    }
    check(clue::read_csv<2, float>(queue, write_file("crlf.csv", content)));
  }
  SUBCASE("Only the header") {
    for (const auto& ending : {"", "\n", "\n\n"}) {
      const auto path = write_file("header.csv", std::string{"x0,x1,weight"} + ending);
      auto invalid_call = [&] { clue::read_csv<2, float>(queue, path); };
      CHECK_THROWS_AS(invalid_call(), std::runtime_error);
    }
  }
}

TEST_CASE("Test read_output with different file endings") {
  auto queue = clue::get_queue(0u);

  const std::string content = "x0,x1,weight,cluster_ids\n1.,2.,3.,0\n4.,5.,6.,-1";
  for (const auto& ending : {std::string{}, std::string{"\n"}, std::string{"\n\n"}}) {
    auto points = clue::read_output<2, float>(queue, write_file("output.csv", content + ending));
    REQUIRE(points.size() == 2);
    CHECK(points.weights()[1] == 6.f);
    CHECK(points.clusterIndexes()[0] == 0);
    CHECK(points.clusterIndexes()[1] == -1);
  }
}
