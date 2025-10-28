
#pragma once

#include <limits>
#include <numeric>
#include <span>

namespace clue::detail {

  inline auto compute_nclusters(std::span<const int> cluster_indexes) {
    return std::reduce(cluster_indexes.begin(),
                       cluster_indexes.end(),
                       std::numeric_limits<int>::lowest(),
                       [](int a, int b) { return std::max(a, b); }) +
           1;
  }

}  // namespace clue::detail
