
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace clue::detail {

  template <concepts::queue TQueue,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>()))>
  void setup_followers(TQueue& queue, std::optional<Followers<TDev>>& followers, int32_t n_points) {
    if (!followers.has_value()) {
      followers = std::make_optional<Followers<TDev>>(n_points, queue);
    }

    if (!(followers->extents() >= n_points)) {
      followers->initialize(n_points, queue);
    } else {
      followers->reset(n_points);
    }
  }

}  // namespace clue::detail
