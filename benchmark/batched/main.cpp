
#include "CLUEstering/CLUEstering.hpp"
#include "utils/generation.hpp"
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <vector>

static void BM_SingleEvents(benchmark::State& state) {
  auto queue = clue::get_queue(0u);

  std::vector<clue::PointsHost<2>> host_points;
  std::ranges::transform(std::views::iota(0u) | std::views::take(1000),
                         std::back_inserter(host_points),
                         [&](const auto i) {
                           return clue::read_csv<2>(
                               queue, "../../data/small_event_" + std::to_string(i) + ".csv");
                         });
  clue::PointsDevice<2> d_points(queue, host_points[0].size());
  const auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;

  for (auto _ : state) {
    for (auto& h_points : host_points) {
      clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
      algo.make_clusters(queue, h_points, d_points);
    }
  }
}

static void BM_Batched(benchmark::State& state) {
  auto queue = clue::get_queue(0u);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../data/small_events_batch.csv");
  const size_t n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);
  const auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
  std::vector<std::size_t> batch_event_sizes(1000, n_points / 1000);

  for (auto _ : state) {
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points, d_points, batch_event_sizes);
  }
}

BENCHMARK(BM_SingleEvents)->Iterations(100);
BENCHMARK(BM_Batched)->Iterations(100);
BENCHMARK_MAIN();
