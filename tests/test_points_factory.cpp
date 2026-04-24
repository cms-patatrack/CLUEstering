
#include "CLUEstering/CLUEstering.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) or defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED) or \
    defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)

namespace {

  template <std::size_t Ndim>
  void fill_coords(std::vector<float>& coords, std::size_t size) {
    for (auto dim = 0u; dim < Ndim; ++dim) {
      for (auto i = 0u; i < size; ++i) {
        coords[dim * size + i] = static_cast<float>(dim * 1000 + i);
      }
    }
  }

}  // namespace

TEST_CASE("make_clustered_points via queue from raw pointers") {
  auto queue = clue::get_queue(0u);
  const auto size = 100u;

  std::vector<float> coords(2 * size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  fill_coords<2>(coords, size);
  for (auto i = 0u; i < size; ++i) {
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      queue, size, coords.data(), weights.data(), cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points via queue from spans") {
  auto queue = clue::get_queue(0u);
  const auto size = 100u;

  std::vector<float> coords(2 * size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  fill_coords<2>(coords, size);
  for (auto i = 0u; i < size; ++i) {
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      queue,
      std::span<const float>(coords.data(), coords.size()),
      std::span<const float>(weights.data(), weights.size()),
      std::span<const int>(cluster_indexes.data(), cluster_indexes.size()));

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points via device from raw pointers") {
  const auto device = clue::get_device(0u);
  const auto size = 100u;

  std::vector<float> coords(2 * size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  fill_coords<2>(coords, size);
  for (auto i = 0u; i < size; ++i) {
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      device, size, coords.data(), weights.data(), cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points via device from spans") {
  const auto device = clue::get_device(0u);
  const auto size = 100u;

  std::vector<float> coords(2 * size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  fill_coords<2>(coords, size);
  for (auto i = 0u; i < size; ++i) {
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      device,
      std::span<const float>(coords.data(), coords.size()),
      std::span<const float>(weights.data(), weights.size()),
      std::span<const int>(cluster_indexes.data(), cluster_indexes.size()));

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

// TODO: wait for cluster properties getters to be implemented for device points
// TEST_CASE("n_clusters") {
//   auto queue = clue::get_queue(0u);
//   const auto size = 100u;

//   std::vector<float> coords(2 * size);
//   std::vector<float> weights(size, 1.f);
//   std::vector<int> cluster_indexes(size);

//   fill_coords<2>(coords, size);
//   for (auto i = 0u; i < size; ++i) {
//     cluster_indexes[i] = static_cast<int>(i / 10u);
//   }

//   auto points = clue::make_clustered_points<2>(
//       queue, coords.data(), weights.data(), cluster_indexes.data(), size);

//   const auto nc = points.n_clusters();
//   CHECK(nc == 10u);

//   CHECK(points.n_clusters() == nc);
// }

// TEST_CASE("cluster_sizes") {
//   auto queue = clue::get_queue(0u);
//   const auto size = 100u;

//   std::vector<float> coords(2 * size);
//   std::vector<float> weights(size, 1.f);
//   std::vector<int> cluster_indexes(size);

//   fill_coords<2>(coords, size);
//   for (auto i = 0u; i < size; ++i) {
//     cluster_indexes[i] = static_cast<int>(i / 10u);
//   }

//   auto points = clue::make_clustered_points<2>(
//       queue, coords.data(), weights.data(), cluster_indexes.data(), size);

//   const auto& sizes = points.cluster_sizes();
//   REQUIRE(sizes.size() == 10u);
//   std::ranges::for_each(sizes, [](auto s) { CHECK(s == 10u); });
// }

// TEST_CASE("clusters association map") {
//   auto queue = clue::get_queue(0u);
//   const auto size = 60u;

//   std::vector<float> coords(2 * size);
//   std::vector<float> weights(size, 1.f);
//   std::vector<int> cluster_indexes(size);

//   fill_coords<2>(coords, size);
//   for (auto i = 0u; i < size; ++i) {
//     cluster_indexes[i] = static_cast<int>(i / 20u);
//   }

//   auto points = clue::make_clustered_points<2>(
//       queue, coords.data(), weights.data(), cluster_indexes.data(), size);

//   const auto& map = points.clusters();
//   CHECK(map.size() == points.n_clusters());
//   for (auto c = 0u; c < points.n_clusters(); ++c) {
//     CHECK(map.count(static_cast<int>(c)) == 20u);
//   }
// }

// TEST_CASE("cluster_properties") {
//   auto queue = clue::get_queue(0u);
//   const auto size = 60u;

//   std::vector<float> coords(2 * size);
//   std::vector<float> weights(size, 1.f);
//   std::vector<int> cluster_indexes(size);

//   fill_coords<2>(coords, size);
//   for (auto i = 0u; i < size; ++i) {
//     cluster_indexes[i] = static_cast<int>(i / 20u);
//   }

//   auto points = clue::make_clustered_points<2>(
//       queue, coords.data(), weights.data(), cluster_indexes.data(), size);

//   const auto& props = points.cluster_properties();
//   CHECK(props.n_clusters() == 3u);
//   CHECK(props.cluster_sizes().size() == 3u);
// }

// TEST_CASE("cluster properties with outliers") {
//   auto queue = clue::get_queue(0u);
//   const auto size = 50u;

//   std::vector<float> coords(2 * size);
//   std::vector<float> weights(size, 1.f);
//   std::vector<int> cluster_indexes(size);

//   fill_coords<2>(coords, size);
//   for (auto i = 0u; i < size; ++i) {
//     cluster_indexes[i] = (i < 30u) ? static_cast<int>(i / 10u) : -1;
//   }

//   auto points = clue::make_clustered_points<2>(
//       queue, coords.data(), weights.data(), cluster_indexes.data(), size);

//   CHECK(points.clustered());
//   CHECK(points.n_clusters() == 3u);

//   const auto& sizes = points.cluster_sizes();
//   REQUIRE(sizes.size() == 3u);
//   std::ranges::for_each(sizes, [](auto s) { CHECK(s == 10u); });
// }

TEST_CASE("make_clustered_points with 1D points") {
  auto queue = clue::get_queue(0u);
  const auto size = 60u;

  std::vector<float> coords(size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  std::iota(coords.begin(), coords.end(), 0.f);
  for (auto i = 0u; i < size; ++i) {
    cluster_indexes[i] = static_cast<int>(i / 20u);
  }

  auto points = clue::make_clustered_points<1>(
      queue, size, coords.data(), weights.data(), cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.n_clusters() == 3u);
}

TEST_CASE("make_clustered_points with 3D points") {
  auto queue = clue::get_queue(0u);
  const auto size = 60u;

  std::vector<float> coords(3 * size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  fill_coords<3>(coords, size);
  for (auto i = 0u; i < size; ++i) {
    cluster_indexes[i] = static_cast<int>(i / 20u);
  }

  auto points = clue::make_clustered_points<3>(
      queue, size, coords.data(), weights.data(), cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.n_clusters() == 3u);
}

TEST_CASE("make_clustered_points with double precision") {
  auto queue = clue::get_queue(0u);
  const auto size = 100u;

  std::vector<double> coords(2 * size);
  std::vector<double> weights(size, 1.0);
  std::vector<int> cluster_indexes(size);

  for (auto dim = 0u; dim < 2u; ++dim) {
    for (auto i = 0u; i < size; ++i) {
      coords[dim * size + i] = static_cast<double>(dim * 1000u + i);
    }
  }
  for (auto i = 0u; i < size; ++i) {
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      queue, size, coords.data(), weights.data(), cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.n_clusters() == 4u);

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points via queue from per-dimension raw pointers") {
  auto queue = clue::get_queue(0u);
  const auto size = 100u;

  std::vector<float> x_coords(size);
  std::vector<float> y_coords(size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  for (auto i = 0u; i < size; ++i) {
    x_coords[i] = static_cast<float>(i);
    y_coords[i] = static_cast<float>(1000u + i);
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      queue, size, x_coords.data(), y_coords.data(), weights.data(), cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points via queue from per-dimension spans") {
  auto queue = clue::get_queue(0u);
  const auto size = 100u;

  std::vector<float> x_coords(size);
  std::vector<float> y_coords(size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  for (auto i = 0u; i < size; ++i) {
    x_coords[i] = static_cast<float>(i);
    y_coords[i] = static_cast<float>(1000u + i);
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      queue,
      std::span<const float>(x_coords.data(), x_coords.size()),
      std::span<const float>(y_coords.data(), y_coords.size()),
      std::span<const float>(weights.data(), weights.size()),
      std::span<const int>(cluster_indexes.data(), cluster_indexes.size()));

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points via device from per-dimension raw pointers") {
  const auto device = clue::get_device(0u);
  const auto size = 100u;

  std::vector<float> x_coords(size);
  std::vector<float> y_coords(size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  for (auto i = 0u; i < size; ++i) {
    x_coords[i] = static_cast<float>(i);
    y_coords[i] = static_cast<float>(1000u + i);
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      device, size, x_coords.data(), y_coords.data(), weights.data(), cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points via device from per-dimension spans") {
  const auto device = clue::get_device(0u);
  const auto size = 100u;

  std::vector<float> x_coords(size);
  std::vector<float> y_coords(size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  for (auto i = 0u; i < size; ++i) {
    x_coords[i] = static_cast<float>(i);
    y_coords[i] = static_cast<float>(1000u + i);
    cluster_indexes[i] = static_cast<int>(i / 25u);
  }

  auto points = clue::make_clustered_points<2>(
      device,
      std::span<const float>(x_coords.data(), x_coords.size()),
      std::span<const float>(y_coords.data(), y_coords.size()),
      std::span<const float>(weights.data(), weights.size()),
      std::span<const int>(cluster_indexes.data(), cluster_indexes.size()));

  CHECK(points.clustered());
  CHECK(points.size() == static_cast<int>(size));

  auto ci = points.clusterIndexes();
  CHECK(std::ranges::equal(ci, cluster_indexes));
}

TEST_CASE("make_clustered_points per-dimension pointers with 3D points") {
  auto queue = clue::get_queue(0u);
  const auto size = 60u;

  std::vector<float> x_coords(size);
  std::vector<float> y_coords(size);
  std::vector<float> z_coords(size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size);

  for (auto i = 0u; i < size; ++i) {
    x_coords[i] = static_cast<float>(i);
    y_coords[i] = static_cast<float>(1000u + i);
    z_coords[i] = static_cast<float>(2000u + i);
    cluster_indexes[i] = static_cast<int>(i / 20u);
  }

  auto points = clue::make_clustered_points<3>(queue,
                                               size,
                                               x_coords.data(),
                                               y_coords.data(),
                                               z_coords.data(),
                                               weights.data(),
                                               cluster_indexes.data());

  CHECK(points.clustered());
  CHECK(points.n_clusters() == 3u);
}

TEST_CASE("coords alias the original per-dimension buffers") {
  auto queue = clue::get_queue(0u);
  const auto size = 50u;

  std::vector<float> x_coords(size);
  std::vector<float> y_coords(size);
  std::vector<float> weights(size, 1.f);
  std::vector<int> cluster_indexes(size, 0);

  for (auto i = 0u; i < size; ++i) {
    x_coords[i] = static_cast<float>(i);
    y_coords[i] = static_cast<float>(1000u + i);
  }

  auto points = clue::make_clustered_points<2>(
      queue, size, x_coords.data(), y_coords.data(), weights.data(), cluster_indexes.data());

  auto x = points.coords(0u);
  auto y = points.coords(1u);
  for (auto i = 0u; i < size; ++i) {
    CHECK(x[i] == static_cast<float>(i));
    CHECK(y[i] == static_cast<float>(1000u + i));
  }
}

TEST_CASE("coords and weights alias the original buffers") {
  auto queue = clue::get_queue(0u);
  const auto size = 50u;

  std::vector<float> coords(2 * size);
  std::vector<float> weights(size);
  std::vector<int> cluster_indexes(size, 0);

  fill_coords<2>(coords, size);
  std::iota(weights.begin(), weights.end(), 1.f);

  auto points = clue::make_clustered_points<2>(
      queue, size, coords.data(), weights.data(), cluster_indexes.data());

  for (auto dim = 0u; dim < 2u; ++dim) {
    auto c = points.coords(dim);
    for (auto i = 0u; i < size; ++i) {
      CHECK(c[i] == static_cast<float>(dim * 1000u + i));
    }
  }

  auto w = points.weights();
  for (auto i = 0u; i < size; ++i) {
    CHECK(w[i] == static_cast<float>(i + 1u));
  }
}

#endif
