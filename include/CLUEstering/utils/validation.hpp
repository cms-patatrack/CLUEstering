
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"
#include "CLUEstering/utils/detail/get_clusters.hpp"
#include <algorithm>
#include <concepts>
#include <ranges>
#include <span>
#include <vector>

namespace clue {

  template <std::size_t Ndim, std::floating_point TData = float>
  inline bool validate_results(PointsHost<Ndim, TData>& results, PointsHost<Ndim, TData>& truth) {
    auto result_clusters_sizes = results.cluster_sizes();
    auto truth_clusters_sizes = truth.cluster_sizes();
    std::ranges::sort(result_clusters_sizes);
    std::ranges::sort(truth_clusters_sizes);

    bool compare_nclusters = results.n_clusters() == truth.n_clusters();
    bool compare_clusters_size = std::ranges::equal(result_clusters_sizes, truth_clusters_sizes);

    return compare_nclusters && compare_clusters_size;
  }

}  // namespace clue
