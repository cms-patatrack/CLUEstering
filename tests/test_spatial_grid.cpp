#include "CLUEstering/CLUEstering.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

namespace {

  // Bin of a coordinate, computed as documented in SpatialGrid
  template <typename TData>
  int32_t expected_bin(TData coord, TData min, TData cell_size, int32_t n_bins) {
    auto bin = static_cast<int32_t>((coord - min) / cell_size);
    return std::clamp(bin, 0, n_bins - 1);
  }

  // Row-major global bin, the last dimension varies fastest
  template <std::size_t Ndim, typename TData, typename TDev>
  int32_t expected_global_bin(const clue::PointsHost<Ndim, float>& points,
                              const clue::SpatialGrid<Ndim, TData, TDev>& grid,
                              int32_t point,
                              std::size_t event) {
    int32_t global_bin = 0;
    int32_t stride = 1;
    for (int dim = Ndim - 1; dim >= 0; --dim) {
      const auto bin = expected_bin(
          points.coords(dim)[point], grid.min[dim], grid.cell_size[dim], grid.n_cells[dim]);
      global_bin += bin * stride;
      stride *= grid.n_cells[dim];
    }
    return global_bin + static_cast<int32_t>(event) * stride;
  }

  template <typename TDev>
  struct HostCells {
    std::vector<int32_t> keys;
    std::vector<int32_t> values;
  };

  // Copy the cells to the host with a plain memcpy of the CSR buffers
  template <std::size_t Ndim, typename TData, typename TDev>
  HostCells<TDev> copy_to_host(clue::Queue& queue,
                               const clue::SpatialGrid<Ndim, TData, TDev>& grid) {
    const auto extents = grid.cells.extents();
    const auto containers = grid.cells.extract();

    HostCells<TDev> host_cells{std::vector<int32_t>(extents.keys + 1),
                               std::vector<int32_t>(extents.values)};
    alpaka::memcpy(queue,
                   alpaka::createView(clue::host, host_cells.keys.data(), extents.keys + 1),
                   alpaka::createView(queue, containers.keys.data(), extents.keys + 1));
    alpaka::memcpy(queue,
                   alpaka::createView(clue::host, host_cells.values.data(), extents.values),
                   alpaka::createView(queue, containers.values.data(), extents.values));
    alpaka::wait(queue);
    return host_cells;
  }

  int32_t product(const auto& values) {
    return std::accumulate(values.begin(), values.end(), int32_t{1}, std::multiplies<>{});
  }

}  // namespace

TEST_CASE("Test getSpatialGrid before clustering") {
  clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);
  CHECK_THROWS_AS(algo.getSpatialGrid(), std::runtime_error);
}

TEST_CASE("Test spatial grid of a single run") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
  const auto n_points = h_points.size();

  clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);
  algo.make_clusters(queue, h_points);

  const auto grid = algo.getSpatialGrid();
  CHECK(grid.batch_size == 1);
  CHECK(grid.cells.extents().keys == static_cast<std::size_t>(product(grid.n_cells)));
  CHECK(grid.cells.extents().values == static_cast<std::size_t>(n_points));
  for (auto dim = 0u; dim != 2; ++dim) {
    CHECK_FALSE(grid.periodic[dim]);
    CHECK(grid.min[dim] < grid.max[dim]);
    CHECK(grid.n_cells[dim] >= 1);
    CHECK(grid.cell_size[dim] ==
          doctest::Approx((grid.max[dim] - grid.min[dim]) / grid.n_cells[dim]));
  }

  const auto host_cells = copy_to_host(queue, grid);
  CHECK(host_cells.keys.front() == 0);
  CHECK(host_cells.keys.back() == n_points);

  // Every point is in exactly one cell, the one given by the exported geometry
  std::vector<int32_t> seen(n_points, 0);
  for (auto cell = 0u; cell + 1 < host_cells.keys.size(); ++cell) {
    for (auto i = host_cells.keys[cell]; i != host_cells.keys[cell + 1]; ++i) {
      const auto point = host_cells.values[i];
      ++seen[point];
      CHECK(expected_global_bin(h_points, grid, point, 0) == static_cast<int32_t>(cell));
    }
  }
  CHECK(std::ranges::all_of(seen, [](auto count) { return count == 1; }));
}

TEST_CASE("Test spatial grid follow the user extremes") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);

  clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);
  const std::pair<float, float> extremes[2] = {{-100.f, 100.f}, {-50.f, 50.f}};
  algo.setExtremes(extremes);
  algo.make_clusters(queue, h_points);

  const auto grid = algo.getSpatialGrid();
  CHECK(grid.min[0] == -100.f);
  CHECK(grid.max[0] == 100.f);
  CHECK(grid.min[1] == -50.f);
  CHECK(grid.max[1] == 50.f);

  // Changing the extremes after the run does not change the description of the built grid
  const std::pair<float, float> other_extremes[2] = {{0.f, 1.f}, {0.f, 1.f}};
  algo.setExtremes(other_extremes);
  const auto same_grid = algo.getSpatialGrid();
  CHECK(same_grid.min == grid.min);
  CHECK(same_grid.max == grid.max);
}

TEST_CASE("Test spatial grid of a periodic run") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);

  clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);
  algo.setWrappedCoordinates(0, 1);
  const float period = 2.f * std::numbers::pi_v<float>;
  const clue::PeriodicEuclideanMetric<2, float> metric{0.f, period};
  algo.make_clusters(queue, h_points, metric);

  const auto grid = algo.getSpatialGrid();
  CHECK_FALSE(grid.periodic[0]);
  CHECK(grid.periodic[1]);
  CHECK(grid.min[1] == 0.f);
  CHECK(grid.max[1] == doctest::Approx(period));
}

TEST_CASE("Test spatial grid of a batched run") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/batched_data_1024.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
  const auto n_points = h_points.size();

  clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);
  const std::size_t n_events = 10;
  const uint32_t event_size = 1024;
  std::vector<uint32_t> event_sizes(n_events, event_size);
  algo.make_clusters(queue, h_points, event_sizes);

  const auto grid = algo.getSpatialGrid();
  const auto n_cells_total = product(grid.n_cells);
  CHECK(grid.batch_size == n_events);
  CHECK(grid.cells.extents().keys == n_cells_total * n_events);
  CHECK(grid.cells.extents().values == static_cast<std::size_t>(n_points));

  const auto host_cells = copy_to_host(queue, grid);
  for (auto cell = 0u; cell + 1 < host_cells.keys.size(); ++cell) {
    for (auto i = host_cells.keys[cell]; i != host_cells.keys[cell + 1]; ++i) {
      const auto point = host_cells.values[i];
      const auto event = static_cast<std::size_t>(point) / event_size;
      CHECK(expected_global_bin(h_points, grid, point, event) == static_cast<int32_t>(cell));
    }
  }
}
