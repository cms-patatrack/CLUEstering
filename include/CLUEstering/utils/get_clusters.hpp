
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

namespace clue {

  template <std::size_t Ndim>
  inline host_associator get_clusters(const PointsHost<Ndim>& points);

  template <concepts::queue TQueue, std::size_t Ndim>
  inline host_associator get_clusters(TQueue& queue, const PointsDevice<Ndim>& points);

}  // namespace clue

#include "CLUEstering/utils/detail/get_clusters.hpp"
