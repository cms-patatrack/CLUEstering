
#include <algorithm>
#include <span>
#include <vector>

namespace clue {

  int compute_nclusters(std::span<int> cluster_ids) {
    return *std::ranges::max_element(cluster_ids) + 1;
  }

  std::vector<std::vector<int>> compute_clusters_points(
      std::span<int> cluster_ids) {
    const auto nclusters = compute_nclusters(cluster_ids);
    std::vector<std::vector<int>> clusters_points(nclusters);

    std::for_each(
        cluster_ids.begin(), cluster_ids.end(), [&, i = 0](auto cluster_id) mutable {
		  if (cluster_id > -1)
			clusters_points[cluster_id].push_back(i++);
        });
    return clusters_points;
  }

  std::vector<int> compute_clusters_size(std::span<int> cluster_ids) {
    const auto nclusters = compute_nclusters(cluster_ids);
    const auto clusters_points = compute_clusters_points(cluster_ids);

    std::vector<int> clusters(nclusters);
    std::ranges::transform(clusters_points, clusters.begin(), [&](const auto& cluster) {
      return std::accumulate(cluster.begin(), cluster.end(), 0u);
    });
    return clusters;
  }

  bool validate_results(const std::span<int> cluster_ids, const std::span<int> truth) {
    bool compare_nclusters = compute_nclusters(cluster_ids) == compute_nclusters(truth);
    bool compare_clusters_size =
        compute_clusters_size(cluster_ids) == compute_clusters_size(truth);
    return compare_nclusters && compare_clusters_size;
  }

}  // namespace clue
