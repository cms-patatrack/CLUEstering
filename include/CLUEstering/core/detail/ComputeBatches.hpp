
#pragma once

#include <cstddef>
#include <cmath>

namespace clue::detail {

  auto compute_batches(std::size_t n_points, std::size_t batch_size) {
    if (batch_size == 0)
      return 1ul;

    return static_cast<std::size_t>(
        std::ceil(static_cast<float>(n_points) / static_cast<float>(batch_size)));
  }

}  // namespace clue::detail
