
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"

namespace clue {

  inline host_associator get_clusters(std::span<const int> cluster_ids) {
    return internal::make_associator(cluster_ids, cluster_ids.size());
  }

  template <std::size_t Ndim>
  inline host_associator get_clusters(const PointsHost<Ndim>& points) {
    return get_clusters(points.clusterIndexes(), static_cast<int32_t>(points.size()));
  }

}  // namespace clue
