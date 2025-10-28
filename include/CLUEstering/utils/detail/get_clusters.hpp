
#pragma once

#include "CLUEstering/utils/get_clusters.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"
#include <span>

namespace clue {
  namespace detail {

    inline auto get_clusters(std::span<const int> cluster_ids) {
      return internal::make_associator(cluster_ids, static_cast<int32_t>(cluster_ids.size()));
    }

    template <concepts::queue TQueue>
    inline auto get_clusters(TQueue& queue, std::span<const int> cluster_ids) {
      return internal::make_associator(
          queue, cluster_ids, static_cast<int32_t>(cluster_ids.size()));
    }

  }  // namespace detail

  template <std::size_t Ndim>
  inline auto get_clusters(const PointsHost<Ndim>& points) {
    assert(points.clustered());
    return detail::get_clusters(points.clusterIndexes());
  }

  template <concepts::queue TQueue, std::size_t Ndim>
  inline auto get_clusters(TQueue& queue, const PointsDevice<Ndim>& points) {
    assert(points.clustered());
    return detail::get_clusters(queue, points.clusterIndexes());
  }

}  // namespace clue
