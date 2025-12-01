
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

  auto compute_batch_blocks(std::size_t max_batch_item_size, std::size_t block_size) {
    return static_cast<std::size_t>(
        std::ceil(static_cast<float>(max_batch_item_size) / static_cast<float>(block_size)));
  }

}  // namespace clue::detail
