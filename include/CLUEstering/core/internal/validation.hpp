
#pragma once

#include <algorithm>
#include <ranges>
#include <span>
#include <vector>

namespace clue::internal {

  inline int compute_nclusters(std::span<const int> cluster_ids) {
    return std::reduce(cluster_ids.begin(),
                       cluster_ids.end(),
                       std::numeric_limits<int>::lowest(),
                       [](int a, int b) { return std::max(a, b); }) +
           1;
  }

  inline std::vector<std::vector<int>> compute_clusters_points(std::span<const int> cluster_ids) {
    const auto nclusters = compute_nclusters(cluster_ids);
    std::vector<std::vector<int>> clusters_points(nclusters);

    std::for_each(cluster_ids.begin(), cluster_ids.end(), [&, i = 0](auto cluster_id) mutable {
      if (cluster_id > -1)
        clusters_points[cluster_id].push_back(i++);
    });

    return clusters_points;
  }

  inline std::vector<int> compute_clusters_size(std::span<const int> cluster_ids) {
    const auto nclusters = compute_nclusters(cluster_ids);
    const auto clusters_points = compute_clusters_points(cluster_ids);

    std::vector<int> clusters(nclusters);
    std::ranges::transform(
        clusters_points, clusters.begin(), [&](const auto& cluster) { return cluster.size(); });
    return clusters;
  }

}  // namespace clue::internal
