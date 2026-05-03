
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"
#include "utils.hpp"

#include <numbers>
#include <random>
#include <ranges>

#include <fmt/core.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("Test clustering on benchmarking datasets") {
  auto range = std::make_pair(10, 14);
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const float dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      for (auto i = range.first; i < range.second; ++i) {
        const auto test_file_path =
            std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
        auto input_data = test::read_csv_combined<float>(test_file_path);
        const auto n_points = input_data.n_points;
        auto h_points = clue::PointsHost<2>(queue, n_points, input_data.data, input_data.labels);

        algo.make_clusters(queue, h_points);
        CHECK(clue::silhouette(h_points) >= 0.9f);
      }
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto dc = 1.5, rhoc = 10., outlier = 1.5;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      for (auto i = range.first; i < range.second; ++i) {
        const auto test_file_path =
            std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
        auto input_data = test::read_csv_combined<double>(test_file_path);
        const auto n_points = input_data.n_points;
        auto h_points =
            clue::PointsHost<2, double>(queue, n_points, input_data.data, input_data.labels);

        algo.make_clusters(queue, h_points);
        CHECK(clue::silhouette(h_points) >= 0.9f);
      }
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const float dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      for (auto i = range.first; i < range.second; ++i) {
        const auto test_file_path =
            std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
        auto input_data = test::read_csv_separate<float>(test_file_path);
        const auto n_points = input_data.n_points;
        auto h_points = clue::PointsHost<2>(
            queue, n_points, input_data.coords, input_data.weights, input_data.labels);

        algo.make_clusters(queue, h_points);
        CHECK(clue::silhouette(h_points) >= 0.9f);
      }
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto dc = 1.5, rhoc = 10., outlier = 1.5;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      for (auto i = range.first; i < range.second; ++i) {
        const auto test_file_path =
            std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
        auto input_data = test::read_csv_separate<double>(test_file_path);
        const auto n_points = input_data.n_points;
        auto h_points = clue::PointsHost<2, double>(
            queue, n_points, input_data.coords, input_data.weights, input_data.labels);

        algo.make_clusters(queue, h_points);
        CHECK(clue::silhouette(h_points) >= 0.9f);
      }
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const float dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      for (auto i = range.first; i < range.second; ++i) {
        const auto test_file_path =
            std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
        auto input_data = test::read_csv_per_dim<float>(test_file_path);
        const auto n_points = input_data.n_points;
        auto h_points = clue::PointsHost<2>(queue,
                                            n_points,
                                            input_data.dims[0],
                                            input_data.dims[1],
                                            input_data.weights,
                                            input_data.labels);

        algo.make_clusters(queue, h_points);
        CHECK(clue::silhouette(h_points) >= 0.9f);
      }
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto dc = 1.5, rhoc = 10., outlier = 1.5;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      for (auto i = range.first; i < range.second; ++i) {
        const auto test_file_path =
            std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, i));
        auto input_data = test::read_csv_per_dim<double>(test_file_path);
        const auto n_points = input_data.n_points;
        auto h_points = clue::PointsHost<2, double>(queue,
                                                    n_points,
                                                    input_data.dims[0],
                                                    input_data.dims[1],
                                                    input_data.weights,
                                                    input_data.labels);

        algo.make_clusters(queue, h_points);
        CHECK(clue::silhouette(h_points) >= 0.9f);
      }
    }
  }
}

#ifndef COVERAGE
TEST_CASE("Test clustering on a large dataset") {
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const float dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, 18));
      auto input_data = test::read_csv_combined<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue, n_points, input_data.data, input_data.labels);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto dc = 1.5, rhoc = 10., outlier = 1.5;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, 18));
      auto input_data = test::read_csv_combined<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points =
          clue::PointsHost<2, double>(queue, n_points, input_data.data, input_data.labels);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const float dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, 18));
      auto input_data = test::read_csv_separate<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto dc = 1.5, rhoc = 10., outlier = 1.5;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, 18));
      auto input_data = test::read_csv_separate<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const float dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, 18));
      auto input_data = test::read_csv_per_dim<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue,
                                          n_points,
                                          input_data.dims[0],
                                          input_data.dims[1],
                                          input_data.weights,
                                          input_data.labels);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto dc = 1.5, rhoc = 10., outlier = 1.5;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      const auto test_file_path =
          std::string(TEST_DATA_DIR) + fmt::format("/data_{}.csv", std::pow(2, 18));
      auto input_data = test::read_csv_per_dim<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(queue,
                                                  n_points,
                                                  input_data.dims[0],
                                                  input_data.dims[1],
                                                  input_data.weights,
                                                  input_data.labels);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
}
#endif

TEST_CASE("Test clustering on aniso dataset") {
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
      auto input_data = test::read_csv_combined<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue, n_points, input_data.data, input_data.labels);

      const float dc = 25.f, rhoc = 5.f, outlier = 23.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      // TODO: use a better metric for anisotropic data
      // like Davies-Bouldin index
      // CHECK(clue::silhouette(h_points) >= 0.9f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
      auto input_data = test::read_csv_combined<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points =
          clue::PointsHost<2, double>(queue, n_points, input_data.data, input_data.labels);

      const auto dc = 25., rhoc = 5., outlier = 23.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      // TODO: use a better metric for anisotropic data
      // like Davies-Bouldin index
      // CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
      auto input_data = test::read_csv_separate<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const float dc = 25.f, rhoc = 5.f, outlier = 23.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      // TODO: use a better metric for anisotropic data
      // like Davies-Bouldin index
      // CHECK(clue::silhouette(h_points) >= 0.9f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
      auto input_data = test::read_csv_separate<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const auto dc = 25., rhoc = 5., outlier = 23.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      // TODO: use a better metric for anisotropic data
      // like Davies-Bouldin index
      // CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
      auto input_data = test::read_csv_per_dim<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue,
                                          n_points,
                                          input_data.dims[0],
                                          input_data.dims[1],
                                          input_data.weights,
                                          input_data.labels);

      const float dc = 25.f, rhoc = 5.f, outlier = 23.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      // TODO: use a better metric for anisotropic data
      // like Davies-Bouldin index
      // CHECK(clue::silhouette(h_points) >= 0.9f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/aniso_1000.csv";
      auto input_data = test::read_csv_per_dim<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(queue,
                                                  n_points,
                                                  input_data.dims[0],
                                                  input_data.dims[1],
                                                  input_data.weights,
                                                  input_data.labels);

      const auto dc = 25., rhoc = 5., outlier = 23.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      // TODO: use a better metric for anisotropic data
      // like Davies-Bouldin index
      // CHECK(clue::silhouette(h_points) >= 0.9f);
    }
  }
}

TEST_CASE("Test clustering on sissa 1000 dataset") {
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
      auto input_data = test::read_csv_combined<float>(test_file_path);

      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue, n_points, input_data.data, input_data.labels);

      const auto dc = 20.f, rhoc = 10.f, outlier = 20.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.5f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
      auto input_data = test::read_csv_combined<double>(test_file_path);

      const auto n_points = input_data.n_points;
      auto h_points =
          clue::PointsHost<2, double>(queue, n_points, input_data.data, input_data.labels);

      const auto dc = 20.f, rhoc = 10.f, outlier = 20.f;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.5f);
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
      auto input_data = test::read_csv_separate<float>(test_file_path);

      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const auto dc = 20.f, rhoc = 10.f, outlier = 20.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.5f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
      auto input_data = test::read_csv_separate<double>(test_file_path);

      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const auto dc = 20., rhoc = 10., outlier = 20.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.5f);
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
      auto input_data = test::read_csv_per_dim<float>(test_file_path);

      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue,
                                          n_points,
                                          input_data.dims[0],
                                          input_data.dims[1],
                                          input_data.weights,
                                          input_data.labels);

      const auto dc = 20.f, rhoc = 10.f, outlier = 20.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.5f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_1000.csv";
      auto input_data = test::read_csv_per_dim<double>(test_file_path);

      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(queue,
                                                  n_points,
                                                  input_data.dims[0],
                                                  input_data.dims[1],
                                                  input_data.weights,
                                                  input_data.labels);

      const auto dc = 20., rhoc = 10., outlier = 20.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.5f);
    }
  }
}

TEST_CASE("Test clustering on sissa 4000 dataset") {
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
      auto input_data = test::read_csv_combined<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue, n_points, input_data.data, input_data.labels);

      const float dc = 20.f, rhoc = 10.f, outlier = 20.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.45f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
      auto input_data = test::read_csv_combined<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points =
          clue::PointsHost<2, double>(queue, n_points, input_data.data, input_data.labels);

      const auto dc = 20., rhoc = 10., outlier = 20.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.45f);
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
      auto input_data = test::read_csv_separate<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const float dc = 20.f, rhoc = 10.f, outlier = 20.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.45f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
      auto input_data = test::read_csv_separate<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const auto dc = 20., rhoc = 10., outlier = 20.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.45f);
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
      auto input_data = test::read_csv_per_dim<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue,
                                          n_points,
                                          input_data.dims[0],
                                          input_data.dims[1],
                                          input_data.weights,
                                          input_data.labels);

      const float dc = 20.f, rhoc = 10.f, outlier = 20.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.45f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/sissa_4000.csv";
      auto input_data = test::read_csv_per_dim<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(queue,
                                                  n_points,
                                                  input_data.dims[0],
                                                  input_data.dims[1],
                                                  input_data.weights,
                                                  input_data.labels);

      const auto dc = 20., rhoc = 10., outlier = 20.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.45f);
    }
  }
}

TEST_CASE("Test clustering on toy detector 1000 dataset") {
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
      auto input_data = test::read_csv_combined<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue, n_points, input_data.data, input_data.labels);

      const float dc = 4.f, rhoc = 2.5f, outlier = 4.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);

      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
      auto input_data = test::read_csv_combined<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points =
          clue::PointsHost<2, double>(queue, n_points, input_data.data, input_data.labels);

      const auto dc = 4., rhoc = 2.5, outlier = 4.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
      auto input_data = test::read_csv_separate<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const float dc = 4.f, rhoc = 2.5f, outlier = 4.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
      auto input_data = test::read_csv_separate<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const auto dc = 4., rhoc = 2.5, outlier = 4.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
      auto input_data = test::read_csv_per_dim<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue,
                                          n_points,
                                          input_data.dims[0],
                                          input_data.dims[1],
                                          input_data.weights,
                                          input_data.labels);

      const float dc = 4.f, rhoc = 2.5f, outlier = 4.f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
      auto input_data = test::read_csv_per_dim<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(queue,
                                                  n_points,
                                                  input_data.dims[0],
                                                  input_data.dims[1],
                                                  input_data.weights,
                                                  input_data.labels);

      const auto dc = 4., rhoc = 2.5, outlier = 4.;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
  }
}

TEST_CASE("Test clustering on blob dataset") {
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
      auto input_data = test::read_csv_combined<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<3>(queue, n_points, input_data.data, input_data.labels);

      const float dc = 1.f, rhoc = 5.f, outlier = 2.f;
      clue::Clusterer<3> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
      auto input_data = test::read_csv_combined<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points =
          clue::PointsHost<3, double>(queue, n_points, input_data.data, input_data.labels);

      const auto dc = 1., rhoc = 5., outlier = 2.;
      clue::Clusterer<3, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
      auto input_data = test::read_csv_separate<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<3>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const float dc = 1.f, rhoc = 5.f, outlier = 2.f;
      clue::Clusterer<3> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
      auto input_data = test::read_csv_separate<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<3, double>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const auto dc = 1., rhoc = 5., outlier = 2.;
      clue::Clusterer<3, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
      auto input_data = test::read_csv_per_dim<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<3>(queue,
                                          n_points,
                                          input_data.dims[0],
                                          input_data.dims[1],
                                          input_data.dims[2],
                                          input_data.weights,
                                          input_data.labels);

      const float dc = 1.f, rhoc = 5.f, outlier = 2.f;
      clue::Clusterer<3> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/blob.csv";
      auto input_data = test::read_csv_per_dim<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<3, double>(queue,
                                                  n_points,
                                                  input_data.dims[0],
                                                  input_data.dims[1],
                                                  input_data.dims[2],
                                                  input_data.weights,
                                                  input_data.labels);

      const auto dc = 1., rhoc = 5., outlier = 2.;
      clue::Clusterer<3, double> algo(queue, dc, rhoc, outlier);

      algo.make_clusters(queue, h_points);
      CHECK(clue::silhouette(h_points) >= 0.8f);
    }
  }
}

TEST_CASE("Test clustering on data with periodic coordinates") {
  SUBCASE("Test clustering from combined buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
      auto input_data = test::read_csv_combined<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue, n_points, input_data.data, input_data.labels);

      const float dc = .2f, rhoc = 5.f, outlier = .2f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         h_points,
                         clue::metrics::PeriodicEuclidean(
                             std::array<float, 2>{0.f, 2.f * std::numbers::pi_v<float>}));
      CHECK(h_points.n_clusters() == 1);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
      auto input_data = test::read_csv_combined<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points =
          clue::PointsHost<2, double>(queue, n_points, input_data.data, input_data.labels);

      const auto dc = .2, rhoc = 5., outlier = .2;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         h_points,
                         clue::metrics::PeriodicEuclidean<2, double>(
                             std::array<double, 2>{0., 2. * std::numbers::pi_v<double>}));
      CHECK(h_points.n_clusters() == 1);
    }
  }
  SUBCASE("Test clustering from separate input buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
      auto input_data = test::read_csv_separate<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const float dc = .2f, rhoc = 5.f, outlier = .2f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         h_points,
                         clue::metrics::PeriodicEuclidean(
                             std::array<float, 2>{0.f, 2.f * std::numbers::pi_v<float>}));
      CHECK(h_points.n_clusters() == 1);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
      auto input_data = test::read_csv_separate<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(
          queue, n_points, input_data.coords, input_data.weights, input_data.labels);

      const auto dc = .2, rhoc = 5., outlier = .2;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         h_points,
                         clue::metrics::PeriodicEuclidean<2, double>(
                             std::array<double, 2>{0., 2. * std::numbers::pi_v<double>}));
      CHECK(h_points.n_clusters() == 1);
    }
  }
  SUBCASE("Test clustering from separate coordinates buffers") {
    SUBCASE("Clustering with single-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
      auto input_data = test::read_csv_per_dim<float>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2>(queue,
                                          n_points,
                                          input_data.dims[0],
                                          input_data.dims[1],
                                          input_data.weights,
                                          input_data.labels);

      const float dc = .2f, rhoc = 5.f, outlier = .2f;
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         h_points,
                         clue::metrics::PeriodicEuclidean(
                             std::array<float, 2>{0.f, 2.f * std::numbers::pi_v<float>}));
      CHECK(h_points.n_clusters() == 1);
    }
    SUBCASE("Clustering with double-precision float data") {
      auto queue = clue::get_queue(0u);

      const auto test_file_path = std::string(TEST_DATA_DIR) + "/opposite_angles.csv";
      auto input_data = test::read_csv_per_dim<double>(test_file_path);
      const auto n_points = input_data.n_points;
      auto h_points = clue::PointsHost<2, double>(queue,
                                                  n_points,
                                                  input_data.dims[0],
                                                  input_data.dims[1],
                                                  input_data.weights,
                                                  input_data.labels);

      const auto dc = .2, rhoc = 5., outlier = .2;
      clue::Clusterer<2, double> algo(queue, dc, rhoc, outlier);

      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         h_points,
                         clue::metrics::PeriodicEuclidean<2, double>(
                             std::array<double, 2>{0., 2. * std::numbers::pi_v<double>}));
      CHECK(h_points.n_clusters() == 1);
    }
  }
}

TEST_CASE("Test clustering from constant host points") {
  auto queue = clue::get_queue(0u);

  SUBCASE("single-precision floating point data") {
    std::mt19937 gen;
    std::normal_distribution<float> dis(0.f, .3f);

    const auto size = 1000u;
    const float dc = .2f, rhoc = 1.f, outlier = .2f;
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
    const auto dc = .2, rhoc = 1., outlier = .2;
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

TEST_CASE("Test non-default local density uncertainty") {
  std::mt19937 gen;
  std::normal_distribution<double> dis(0., .3);

  const auto size = 1000u;
  const auto dc = .2f, rhoc = 1.f, outlier = .2f;
  auto queue = clue::get_queue(0u);
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  auto input = clue::make_host_buffer<float[]>(queue, 3 * size);
  auto output = clue::make_host_buffer<int[]>(queue, size);
  std::generate(input.data(), input.data() + 3 * size, [&] { return dis(gen); });
  clue::PointsHost<2> h_points(queue, size, input.data(), output.data());

  SUBCASE("Test interface for host points") {
    SUBCASE("Clustering with default density uncertainty") {
      std::vector<float> density_uncertainty(size, 1.f);
      h_points.set_density_uncertainty(density_uncertainty);
      algo.make_clusters(queue, h_points);
      CHECK(true);
    }
    SUBCASE("Clustering with non-default density uncertainty") {
      std::vector<float> density_uncertainty(size, 5.f);
      h_points.set_density_uncertainty(density_uncertainty);
      algo.make_clusters(queue, h_points);
      CHECK(true);
    }
    SUBCASE("Clustering with variable density uncertainty") {
      std::vector<float> density_uncertainty(size);
      std::generate(
          density_uncertainty.begin(), density_uncertainty.end(), [&] { return dis(gen) + 1.f; });
      h_points.set_density_uncertainty(density_uncertainty);
      algo.make_clusters(queue, h_points);
      CHECK(true);
    }
  }
  SUBCASE("Test interface for device points") {
    auto d_points = clue::PointsDevice<2>(queue, size);
    clue::copyToDevice(queue, d_points, h_points);

    SUBCASE("Clustering with default density uncertainty") {
      std::vector<float> density_uncertainty(size, 1.f);

      auto d_density_uncertainty = clue::make_device_buffer<float[]>(queue, size);
      alpaka::memcpy(
          queue, d_density_uncertainty, clue::make_host_view(density_uncertainty.data(), size));
      alpaka::wait(queue);

      d_points.set_density_uncertainty(std::span<float>{d_density_uncertainty.data(), size});
      algo.make_clusters(queue, d_points);
      CHECK(true);
    }
    SUBCASE("Clustering with non-default density uncertainty") {
      std::vector<float> density_uncertainty(size, 5.f);

      auto d_density_uncertainty = clue::make_device_buffer<float[]>(queue, size);
      alpaka::memcpy(
          queue, d_density_uncertainty, clue::make_host_view(density_uncertainty.data(), size));
      alpaka::wait(queue);

      d_points.set_density_uncertainty(std::span<float>{d_density_uncertainty.data(), size});
      algo.make_clusters(queue, d_points);
      CHECK(true);
    }
    SUBCASE("Clustering with variable density uncertainty") {
      std::vector<float> density_uncertainty(size);
      std::generate(
          density_uncertainty.begin(), density_uncertainty.end(), [&] { return dis(gen) + 1.f; });

      auto d_density_uncertainty = clue::make_device_buffer<float[]>(queue, size);
      alpaka::memcpy(
          queue, d_density_uncertainty, clue::make_host_view(density_uncertainty.data(), size));
      alpaka::wait(queue);

      d_points.set_density_uncertainty(std::span<float>{d_density_uncertainty.data(), size});
      algo.make_clusters(queue, d_points);
      CHECK(true);
    }
  }
}
