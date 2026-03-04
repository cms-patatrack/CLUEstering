
#include "CLUEstering/CLUEstering.hpp"
#include "utils/generation.hpp"
#include <benchmark/benchmark.h>

#include <cstddef>

template <std::size_t Ndim>
static void BM_clustering(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto queue = clue::get_queue(0u);
    const auto n_points = static_cast<std::size_t>(state.range(0));

    clue::PointsHost<Ndim> h_points(queue, n_points);
    clue::PointsDevice<Ndim> d_points(queue, n_points);
    clue::utils::generateRandomData<Ndim>(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);
    const auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
    state.ResumeTiming();

    clue::Clusterer<Ndim> algo(queue, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points, d_points);
  }
}

BENCHMARK(BM_clustering<2>)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);
BENCHMARK_MAIN();
