
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include <cstddef>
#include <optional>

namespace clue::detail {

  template <concepts::queue TQueue,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>())),
            concepts::allocator TAllocator = DefaultAllocator>
  inline void setup_seeds(TQueue& queue,
                          std::optional<clue::internal::SeedArray<TDev>>& seeds,
                          std::size_t seed_candidates,
                          const TAllocator& allocator = TAllocator{}) {
    if (!seeds.has_value() || seeds->capacity() < seed_candidates) {
      seeds = clue::internal::SeedArray<TDev>(queue, seed_candidates, allocator);
    } else {
      seeds->reset(queue);
    }
    alpaka::wait(queue);
  }

}  // namespace clue::detail
