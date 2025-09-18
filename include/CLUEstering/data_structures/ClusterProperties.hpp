
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/utils/detail/get_clusters.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <optional>
#include <vector>

namespace clue {

  template <std::size_t Ndim>
  class PointsHost;

  class ClusterProperties {
  private:
    host_associator m_clusters_to_points;
    std::vector<std::size_t> m_cluster_sizes;
    std::size_t m_nclusters;

    ClusterProperties() = default;
    ClusterProperties(std::span<const int> cluster_indexes)
        : m_clusters_to_points{detail::get_clusters(cluster_indexes)},
          m_cluster_sizes(m_clusters_to_points.size()),
          m_nclusters{m_clusters_to_points.size()} {
      std::ranges::transform(std::views::iota(0) | std::views::take(m_nclusters),
                             m_cluster_sizes.begin(),
                             [&](int i) { return m_clusters_to_points.count(i); });
    }

  public:
    const auto& cluster_sizes() const { return m_clusters_to_points; }
    const auto& clusters() const { return m_cluster_sizes; }
    const auto& n_clusters() const { return m_nclusters; }

  private:
    template <std::size_t Ndim>
    friend class PointsHost;
  };

}  // namespace clue
