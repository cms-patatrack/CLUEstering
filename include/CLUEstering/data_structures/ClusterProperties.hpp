/// @file ClusterProperties.hpp
/// @brief Defines the ClusterProperties class for accessing the content of the cluster
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/utils/detail/get_clusters.hpp"
#include <ranges>
#include <vector>

namespace clue {

  template <std::size_t Ndim>
  class PointsHost;

  /// @brief The ClusterProperties class provides access to the properties of clusters
  /// such as the number of clusters, the size of each cluster and point associations.
  class ClusterProperties {
  private:
    host_associator m_clusters_to_points;
    std::vector<std::size_t> m_cluster_sizes;
    std::size_t m_nclusters;

    ClusterProperties(std::span<const int> cluster_indexes)
        : m_clusters_to_points{detail::get_clusters(cluster_indexes)},
          m_cluster_sizes(m_clusters_to_points.size()),
          m_nclusters{m_clusters_to_points.size()} {
      std::ranges::transform(std::views::iota(0) | std::views::take(m_nclusters),
                             m_cluster_sizes.begin(),
                             [&](int i) { return m_clusters_to_points.count(i); });
    }

  public:
    /// @brief Returns a vector containing the sizes of each cluster
    ///
    /// @return A vector of containing the sizes of each cluster
    const auto& cluster_sizes() const { return m_clusters_to_points; }
    /// @brief Returns an associator mapping clusters to their associated points
    ///
    /// @return An host_associator mapping clusters to points
    const auto& clusters() const { return m_cluster_sizes; }
    /// @brief Returns the number of clusters
    ///
    /// @return The number of clusters
    const auto& n_clusters() const { return m_nclusters; }

  private:
    template <std::size_t Ndim>
    friend class PointsHost;
  };

}  // namespace clue
