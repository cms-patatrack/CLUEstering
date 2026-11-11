
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/count_if/count_if.hpp"
#include "CLUEstering/utils/get_clusters.hpp"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>

CLUE_NAMESPACE_BEGIN
namespace detail {

  inline auto get_clusters(std::span<const int> cluster_ids) {
    return internal::make_associator(cluster_ids);
  }

  template <concepts::queue TQueue, concepts::allocator TAllocator>
  inline auto get_clusters(TQueue& queue,
                           std::span<const int> cluster_ids,
                           const TAllocator& allocator) {
    return internal::make_associator(queue, cluster_ids, allocator);
  }

}  // namespace detail

template <std::size_t Ndim, std::floating_point TData>
inline auto get_clusters(const PointsHost<Ndim, TData>& points) {
  assert(points.clustered());
  return detail::get_clusters(points.clusterIndexes());
}

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point TData,
          concepts::allocator TAllocator>
inline auto get_clusters(TQueue& queue,
                         const PointsDevice<Ndim, TData>& points,
                         const TAllocator& allocator) {
  assert(points.clustered());
  return detail::get_clusters(queue, points.clusterIndexes(), allocator);
}

CLUE_NAMESPACE_END
