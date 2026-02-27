
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <numbers>
#include <random>
#include <ranges>

#include <fmt/core.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("Test clustering on benchmarking datasets") {
#ifdef COVERAGE
  auto range = std::make_pair(10, 14);
#else
  auto range = std::make_pair(10, 18);
#endif
  SUBCASE("Clustering with single-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    for (auto i = range.first; i < range.second; ++i) {
      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
      clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
      const auto n_points = h_points.size();
      clue::PointsDevice<2> d_points(queue, n_points);

      algo.make_clusters(queue, h_points, d_points);

      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
  SUBCASE("Clustering with double-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto dc{1.5}, rhoc{10.}, outlier{1.5};
    clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

    for (auto i = range.first; i < range.second; ++i) {
      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
      clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
      const auto n_points = h_points.size();
      clue::PointsDevice<2, double> d_points(queue, n_points);

      algo.make_clusters(queue, h_points, d_points);

      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
}

TEST_CASE("Test clustering on aniso dataset") {
  SUBCASE("Clustering with single-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2> d_points(queue, n_points);

    const float dc{25.f}, rhoc{5.f}, outlier{23.f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);
    // TODO: use a better metric for anisotropic data
    // like Davies-Bouldin index
    // CHECK(clue::silhouette(h_points) >= 0.9f);
  }
  SUBCASE("Clustering with double-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
    clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2, double> d_points(queue, n_points);

    const auto dc{25.}, rhoc{5.}, outlier{23.};
    clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);
    // TODO: use a better metric for anisotropic data
    // like Davies-Bouldin index
    // CHECK(clue::silhouette(h_points) >= 0.9f);
  }
}

TEST_CASE("Test clustering on sissa 1000 dataset") {
  SUBCASE("Clustering with single-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2> d_points(queue, n_points);

    const float dc{20.f}, rhoc{10.f}, outlier{20.f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.5f);
  }
  SUBCASE("Clustering with double-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
    clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2, double> d_points(queue, n_points);

    const auto dc{20.}, rhoc{10.}, outlier{20.};
    clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.5f);
  }
}

TEST_CASE("Test clustering on sissa 4000 dataset") {
  SUBCASE("Clustering with single-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2> d_points(queue, n_points);

    const float dc{20.f}, rhoc{10.f}, outlier{20.f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.45f);
  }
  SUBCASE("Clustering with double-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
    clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2, double> d_points(queue, n_points);

    const auto dc{20.}, rhoc{10.}, outlier{20.};
    clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.45f);
  }
}

TEST_CASE("Test clustering on toy detector 1000 dataset") {
  SUBCASE("Clustering with single-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2> d_points(queue, n_points);

    const float dc{4.f}, rhoc{2.5f}, outlier{4.f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.8f);
  }
  SUBCASE("Clustering with double-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
    clue::PointsHost<2, double> h_points = clue::read_csv<2, double>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<2, double> d_points(queue, n_points);

    const auto dc{4.}, rhoc{2.5}, outlier{4.};
    clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.8f);
  }
}

TEST_CASE("Test clustering on blob dataset") {
  SUBCASE("Clustering with single-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
    clue::PointsHost<3> h_points = clue::read_csv<3, float>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<3> d_points(queue, n_points);

    const float dc{1.f}, rhoc{5.f}, outlier{2.f};
    clue::Clusterer<3> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.8f);
  }
  SUBCASE("Clustering with double-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
    clue::PointsHost<3, double> h_points = clue::read_csv<3, double>(queue, test_file_path);
    const auto n_points = h_points.size();
    clue::PointsDevice<3, double> d_points(queue, n_points);

    const auto dc{1.}, rhoc{5.}, outlier{2.};
    clue::Clusterer<3, double> algo(queue, dc, rhoc, outlier);

    algo.make_clusters(queue, h_points, d_points);

    CHECK(clue::silhouette(h_points) >= 0.8f);
  }
}

TEST_CASE("Test clustering on data with periodic coordinates") {
  SUBCASE("Clustering with single-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
    clue::PointsHost<2> points = clue::read_csv<2, float>(queue, test_file_path);
    const float dc{.2f}, rhoc{5.f}, outlier{.2f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

    algo.setWrappedCoordinates(0, 1);
    algo.make_clusters(queue,
                       points,
                       clue::metrics::PeriodicEuclidean(
                           std::array<float, 2>{0.f, 2.f * std::numbers::pi_v<float>}));
    CHECK(points.n_clusters() == 1);
  }
  SUBCASE("Clustering with double-precision float data") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
    clue::PointsHost<2, double> points = clue::read_csv<2, double>(queue, test_file_path);
    const auto dc{.2}, rhoc{5.}, outlier{.2};
    clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

    algo.setWrappedCoordinates(0, 1);
    algo.make_clusters(queue,
                       points,
                       clue::metrics::PeriodicEuclidean<2, double>(
                           std::array<double, 2>{0., 2. * std::numbers::pi_v<double>}));
    CHECK(points.n_clusters() == 1);
  }
}

TEST_CASE("Test clustering from constant host points") {
  auto queue = clue::get_queue(0u);

  SUBCASE("single-precision floating point data") {
    std::mt19937 gen;
    std::normal_distribution<float> dis(0.f, .3f);

    const auto size = 1000u;
    const float dc{.2f}, rhoc{1.f}, outlier{.2f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
    SUBCASE("Two external buffers") {
      auto input = clue::make_host_buffer<float[]>(queue, 3 * size);
      auto output = clue::make_host_buffer<int[]>(queue, size);
      std::generate(input.data(), input.data() + 3 * size, [&] { return dis(gen); });

      clue::ConstPointsHost<2> points_1(queue, size, input.data(), output.data());
      clue::ConstPointsHost<2> points_2(
          queue, size, std::span{input.data(), 3 * size}, std::span{output.data(), size});
      algo.make_clusters(queue, points_1);
      algo.make_clusters(queue, points_2);

      CHECK(true);
    }
    SUBCASE("Three external buffers") {
      auto coords = clue::make_host_buffer<float[]>(queue, 2 * size);
      auto weights = clue::make_host_buffer<float[]>(queue, size);
      auto cluster_ids = clue::make_host_buffer<int[]>(queue, size);
      std::generate(coords.data(), coords.data() + 2 * size, [&] { return dis(gen); });
      std::fill(weights.data(), weights.data() + size, 1.f);

      clue::ConstPointsHost<2> points_1(
          queue, size, coords.data(), weights.data(), cluster_ids.data());
      clue::ConstPointsHost<2> points_2(queue,
                                        size,
                                        std::span{coords.data(), 2 * size},
                                        std::span{weights.data(), size},
                                        std::span{cluster_ids.data(), size});
      algo.make_clusters(queue, points_1);
      algo.make_clusters(queue, points_2);

      CHECK(true);
    }
    SUBCASE("Four external buffers") {
      auto x0 = clue::make_host_buffer<float[]>(queue, size);
      auto x1 = clue::make_host_buffer<float[]>(queue, size);
      auto weights = clue::make_host_buffer<float[]>(queue, size);
      auto cluster_ids = clue::make_host_buffer<int[]>(queue, size);
      std::generate(x0.data(), x0.data() + size, [&] { return dis(gen); });
      std::generate(x1.data(), x1.data() + size, [&] { return dis(gen); });
      std::fill(weights.data(), weights.data() + size, 1.f);

      clue::ConstPointsHost<2> points_1(
          queue, size, x0.data(), x1.data(), weights.data(), cluster_ids.data());
      clue::ConstPointsHost<2> points_2(queue,
                                        size,
                                        std::span{x0.data(), size},
                                        std::span{x1.data(), size},
                                        std::span{weights.data(), size},
                                        std::span{cluster_ids.data(), size});
      algo.make_clusters(queue, points_1);
      algo.make_clusters(queue, points_2);

      CHECK(true);
    }
  }
  SUBCASE("double-precision floating point data") {
    std::mt19937 gen;
    std::normal_distribution<double> dis(0., .3);

    const auto size = 1000u;
    const auto dc{.2}, rhoc{1.}, outlier{.2};
    clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);
    SUBCASE("Two external buffers") {
      auto input = clue::make_host_buffer<double[]>(queue, 3 * size);
      auto output = clue::make_host_buffer<int[]>(queue, size);
      std::generate(input.data(), input.data() + 3 * size, [&] { return dis(gen); });

      clue::ConstPointsHost<2, double> points_1(queue, size, input.data(), output.data());
      clue::ConstPointsHost<2, double> points_2(
          queue, size, std::span{input.data(), 3 * size}, std::span{output.data(), size});
      algo.make_clusters(queue, points_1);
      algo.make_clusters(queue, points_2);

      CHECK(true);
    }
    SUBCASE("Three external buffers") {
      auto coords = clue::make_host_buffer<double[]>(queue, 2 * size);
      auto weights = clue::make_host_buffer<double[]>(queue, size);
      auto cluster_ids = clue::make_host_buffer<int[]>(queue, size);
      std::generate(coords.data(), coords.data() + 2 * size, [&] { return dis(gen); });
      std::fill(weights.data(), weights.data() + size, 1.f);

      clue::ConstPointsHost<2, double> points_1(
          queue, size, coords.data(), weights.data(), cluster_ids.data());
      clue::ConstPointsHost<2, double> points_2(queue,
                                                size,
                                                std::span{coords.data(), 2 * size},
                                                std::span{weights.data(), size},
                                                std::span{cluster_ids.data(), size});
      algo.make_clusters(queue, points_1);
      algo.make_clusters(queue, points_2);

      CHECK(true);
    }
    SUBCASE("Four external buffers") {
      auto x0 = clue::make_host_buffer<double[]>(queue, size);
      auto x1 = clue::make_host_buffer<double[]>(queue, size);
      auto weights = clue::make_host_buffer<double[]>(queue, size);
      auto cluster_ids = clue::make_host_buffer<int[]>(queue, size);
      std::generate(x0.data(), x0.data() + size, [&] { return dis(gen); });
      std::generate(x1.data(), x1.data() + size, [&] { return dis(gen); });
      std::fill(weights.data(), weights.data() + size, 1.f);

      clue::ConstPointsHost<2, double> points_1(
          queue, size, x0.data(), x1.data(), weights.data(), cluster_ids.data());
      clue::ConstPointsHost<2, double> points_2(queue,
                                                size,
                                                std::span{x0.data(), size},
                                                std::span{x1.data(), size},
                                                std::span{weights.data(), size},
                                                std::span{cluster_ids.data(), size});
      algo.make_clusters(queue, points_1);
      algo.make_clusters(queue, points_2);

      CHECK(true);
    }
  }
}
