
#pragma once

#include <algorithm>
#include <ranges>
#include <span>
#include <vector>

namespace clue {

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

  inline bool validate_results(std::span<const int> cluster_ids, std::span<const int> truth) {
    auto result_clusters_sizes = compute_clusters_size(cluster_ids);
    auto truth_clusters_sizes = compute_clusters_size(truth);
    std::ranges::sort(result_clusters_sizes);
    std::ranges::sort(truth_clusters_sizes);

    bool compare_nclusters = compute_nclusters(cluster_ids) == compute_nclusters(truth);
    bool compare_clusters_size = std::ranges::equal(result_clusters_sizes, truth_clusters_sizes);

    return compare_nclusters && compare_clusters_size;
  }

}  // namespace clue
