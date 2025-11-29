
#pragma once

#include "CLUEstering/internal/alpaka/work_division.hpp"
#include <cstddef>

namespace clue::detail {

  auto compute_grid_size(std::int32_t points, std::size_t batch_size) {
    if (batch_size == 0)
      return static_cast<std::uint32_t>(points);

    return (batch_size > 1024) ? clue::divide_up_by(batch_size, 256) : 1u;
  }

}  // namespace clue::detail
