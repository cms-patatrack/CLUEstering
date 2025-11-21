
#include "CLUEstering/CLUEstering.hpp"
#include "utils/generation.hpp"
#include <benchmark/benchmark.h>

#include <cstddef>

static void BM_clustering(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto queue = clue::get_queue(0u);
    const auto n_points = static_cast<std::size_t>(state.range(0));

    clue::PointsHost<2> h_points(queue, n_points);
    clue::PointsDevice<2> d_points(queue, n_points);
    clue::utils::generateRandomData<2>(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);
    const auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
    state.ResumeTiming();

    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points, d_points);
  }
}

BENCHMARK(BM_clustering)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);
BENCHMARK_MAIN();
