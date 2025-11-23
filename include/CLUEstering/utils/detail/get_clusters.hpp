
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/count_if/count_if.hpp"
#include "CLUEstering/utils/get_clusters.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>

namespace clue {
  namespace detail {

    template <typename T>
    struct non_negative {
      ALPAKA_FN_HOST_ACC constexpr auto operator()(T value) const { return value > -1; }
    };

    inline auto get_clusters(std::span<const int> cluster_ids) {
      auto clustered_points = std::ranges::count_if(cluster_ids, non_negative<int>{});
      return internal::make_associator(cluster_ids, static_cast<int>(clustered_points));
    }

    template <concepts::queue TQueue>
    inline auto get_clusters(TQueue& queue, std::span<const int> cluster_ids) {
      auto clustered_points = internal::algorithm::count_if(
          cluster_ids.begin(), cluster_ids.end(), non_negative<int>{});
      return internal::make_associator(queue, cluster_ids, clustered_points);
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
