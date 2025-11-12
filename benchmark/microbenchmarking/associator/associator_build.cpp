
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/CLUEstering.hpp"
#include <benchmark/benchmark.h>
#include <ranges>

static void BM_BuildBinaryAssociatorCPU(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto elements = state.range(0);
    std::vector<int> associations(elements);
    std::ranges::transform(std::views::iota(0, elements),
                           associations.data(),
                           [](auto x) -> int32_t { return x % 2 == 0; });
    state.ResumeTiming();

    volatile auto associator = clue::internal::make_associator(associations, elements);
  }
}

static void BM_BuildBinaryAssociatorAlpaka(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto queue = clue::get_queue(0u);
    const auto elements = state.range(0);
    auto h_associations = clue::make_host_buffer<int[]>(queue, elements);
    std::ranges::transform(std::views::iota(0) | std::views::take(elements),
                           h_associations.data(),
                           [](auto x) -> int32_t { return x % 2 == 0; });
    auto d_associations = clue::make_device_buffer<int[]>(queue, elements);
    alpaka::memcpy(queue, d_associations, h_associations);
    state.ResumeTiming();

    volatile auto associator = clue::internal::make_associator(
        queue, std::span{d_associations.data(), static_cast<std::size_t>(elements)}, elements);
  }
}

BENCHMARK(BM_BuildBinaryAssociatorCPU)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_BuildBinaryAssociatorAlpaka)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

BENCHMARK_MAIN();
