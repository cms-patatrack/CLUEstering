
#include "CLUEstering/CLUEstering.hpp"
#include "utils/generation.hpp"
#include <benchmark/benchmark.h>

#include <cstddef>

inline constexpr auto size = 1 << 17;

static void BM_clustering(benchmark::State& state) {
  auto queue = clue::get_queue(0u);
  clue::PointsHost<2> h_points(queue, size);
  clue::PointsDevice<2> d_points(queue, size);

  for (auto _ : state) {
    state.PauseTiming();
    clue::utils::generateRandomData<2>(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);
    const auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
    state.ResumeTiming();

    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points, d_points);
  }
}

BENCHMARK(BM_clustering)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);
BENCHMARK_MAIN();
