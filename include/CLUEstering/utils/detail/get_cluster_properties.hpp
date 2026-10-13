
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include <limits>
#include <numeric>
#include <span>

namespace clue::inline CLUE_BACKEND::detail {

  inline auto compute_nclusters(std::span<const int> cluster_indexes) {
    return std::reduce(cluster_indexes.begin(),
                       cluster_indexes.end(),
                       std::numeric_limits<int>::lowest(),
                       [](int a, int b) { return std::max(a, b); }) +
           1;
  }

}  // namespace clue::inline CLUE_BACKEND::detail
