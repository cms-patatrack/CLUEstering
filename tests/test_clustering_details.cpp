
#include "CLUEstering/CLUEstering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

namespace {

  // Checks the consistency of the stored details: the nearest higher has a higher (or equal)
  // density, delta is the euclidean distance to it and points without a nearest higher have an
  // infinite delta. The coordinates are read from `coords_source`, because host points created
  // from device points only contain the results of the clustering
  template <std::size_t Ndim, typename TPoints, typename TCoordsSource>
  void check_details(const TPoints& points,
                     const TCoordsSource& coords_source,
                     float max_distance) {
    const auto n_points = static_cast<std::size_t>(points.size());
    const auto rho = points.rho();
    const auto delta = points.delta();
    const auto nearest_higher = points.nearest_higher();
    REQUIRE(rho.size() == n_points);
    REQUIRE(delta.size() == n_points);
    REQUIRE(nearest_higher.size() == n_points);

    for (auto i = 0u; i < n_points; ++i) {
      CHECK(rho[i] >= 0.f);
      const auto nh = nearest_higher[i];
      if (nh == -1) {
        CHECK(std::isinf(delta[i]));
        continue;
      }
      REQUIRE(nh >= 0);
      REQUIRE(static_cast<std::size_t>(nh) < n_points);
      CHECK(rho[nh] >= rho[i]);

      float distance = 0.f;
      for (auto dim = 0u; dim < Ndim; ++dim) {
        const auto diff = coords_source.coords(dim)[i] - coords_source.coords(dim)[nh];
        distance += diff * diff;
      }
      distance = std::sqrt(distance);
      CHECK(delta[i] == doctest::Approx(distance).epsilon(1e-4));
      CHECK(delta[i] <= max_distance);
    }

    // outliers can only follow other outliers
    const auto cluster_ids = points.clusterIndexes();
    for (auto i = 0u; i < n_points; ++i) {
      if (cluster_ids[i] == -1)
        CHECK((nearest_higher[i] == -1 || cluster_ids[nearest_higher[i]] == -1));
    }
  }

  template <std::size_t Ndim, typename TPoints>
  void check_details(const TPoints& points, float max_distance) {
    check_details<Ndim>(points, points, max_distance);
  }

  template <typename TSpan>
  auto to_vector(TSpan span) {
    return std::vector<std::remove_cv_t<typename TSpan::element_type>>(span.begin(), span.end());
  }

}  // namespace

TEST_CASE("Test storing the clustering details") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_4096.csv";
  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(dc, rhoc, outlier);

  SUBCASE("Details are not stored by default") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    CHECK_FALSE(h_points.has_details());
    algo.make_clusters(queue, h_points);
    CHECK_FALSE(h_points.has_details());
    CHECK(h_points.n_clusters() > 0);
  }

  SUBCASE("Storing the details does not change the clustering") {
    clue::PointsHost<2> reference = clue::read_csv<2, float>(queue, test_file_path);
    algo.make_clusters(queue, reference);

    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    h_points.store_details();
    CHECK(h_points.has_details());
    algo.make_clusters(queue, h_points);

    CHECK(h_points.n_clusters() == reference.n_clusters());
    CHECK(h_points.cluster_sizes().size() == reference.cluster_sizes().size());
  }

  SUBCASE("Clustering from host points") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    h_points.store_details();
    algo.make_clusters(queue, h_points);
    check_details<2>(h_points, std::max(dc, outlier));
  }

  SUBCASE("Clustering from host and device points") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    clue::PointsDevice<2> d_points(queue, h_points.size());
    h_points.store_details();
    algo.make_clusters(queue, h_points, d_points);
    check_details<2>(h_points, std::max(dc, outlier));

    // the details must be the same as the ones obtained from host points only
    clue::PointsHost<2> other = clue::read_csv<2, float>(queue, test_file_path);
    other.store_details();
    algo.make_clusters(queue, other);
    CHECK(to_vector(h_points.nearest_higher()) == to_vector(other.nearest_higher()));
    for (auto i = 0; i < h_points.size(); ++i) {
      CHECK(h_points.rho()[i] == doctest::Approx(other.rho()[i]));
      if (std::isinf(other.delta()[i]))
        CHECK(std::isinf(h_points.delta()[i]));
      else
        CHECK(h_points.delta()[i] == doctest::Approx(other.delta()[i]));
    }
  }

  SUBCASE("Copying the device points to new host points") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    h_points.store_details();
    clue::PointsDevice<2> d_points(queue, h_points.size());
    clue::copyToDevice(queue, d_points, h_points);
    algo.make_clusters(queue, d_points);

    auto result = clue::copyToHost(queue, d_points);
    CHECK(result.has_details());
    check_details<2>(result, h_points, std::max(dc, outlier));
  }

  SUBCASE("Device points without details cannot be copied to host points requesting them") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    clue::PointsDevice<2> d_points(queue, h_points.size());
    clue::copyToDevice(queue, d_points, h_points);
    algo.make_clusters(queue, d_points);

    auto result = clue::copyToHost(queue, d_points);
    CHECK_FALSE(result.has_details());

    h_points.store_details();
    CHECK_THROWS_AS(clue::copyToHost(queue, h_points, d_points), std::invalid_argument);
  }

  SUBCASE("Calling store_details more than once") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    h_points.store_details();
    const auto* rho_ptr = h_points.view().m_rho;
    h_points.store_details();
    CHECK(h_points.view().m_rho == rho_ptr);
    algo.make_clusters(queue, h_points);
    check_details<2>(h_points, std::max(dc, outlier));
  }
}

TEST_CASE("Test storing the clustering details of constant host points") {
  auto queue = clue::get_queue(0u);

  std::mt19937 gen;
  std::normal_distribution<float> dis(0.f, .3f);
  const auto size = 1000u;
  const float dc{.2f}, rhoc{1.f}, outlier{.2f};
  clue::Clusterer<2> algo(dc, rhoc, outlier);

  std::vector<float> input(3 * size);
  std::vector<int> output(size);
  std::generate(input.begin(), input.begin() + 2 * size, [&] { return dis(gen); });
  std::fill(input.begin() + 2 * size, input.end(), 1.f);

  clue::ConstPointsHost<2> points(queue, size, input.data(), output.data());
  points.store_details();
  algo.make_clusters(queue, points);
  check_details<2>(points, std::max(dc, outlier));
}

TEST_CASE("Test storing the clustering details with batched clustering") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<2> algo(dc, rhoc, outlier);
  const std::size_t batch_size = 1024;
  std::vector<uint32_t> event_sizes(10, batch_size);
  const auto file_path = std::string(TEST_DATA_DIR) + "/batched_data_1024.csv";

  SUBCASE("From host points") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, file_path);
    h_points.store_details();
    algo.make_clusters(queue, h_points, event_sizes);
    check_details<2>(h_points, std::max(dc, outlier));

    // the nearest higher of each point must belong to the same batch item
    const auto nearest_higher = h_points.nearest_higher();
    for (auto i = 0; i < h_points.size(); ++i) {
      if (nearest_higher[i] != -1)
        CHECK(static_cast<std::size_t>(nearest_higher[i]) / batch_size ==
              static_cast<std::size_t>(i) / batch_size);
    }
  }
  SUBCASE("From host and device points") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, file_path);
    clue::PointsDevice<2> d_points(queue, h_points.size());
    h_points.store_details();
    algo.make_clusters(queue, h_points, d_points, event_sizes);
    check_details<2>(h_points, std::max(dc, outlier));
  }
}
