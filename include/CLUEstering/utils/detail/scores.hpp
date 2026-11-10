
#pragma once

#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

CLUE_NAMESPACE_BEGIN

namespace detail {

  template <std::size_t Ndim, std::floating_point TData, typename DistanceMetric>
  inline auto distance(const clue::PointsView<Ndim, TData>& view,
                       const DistanceMetric& metric,
                       std::size_t i,
                       std::size_t j) -> TData {
    if constexpr (concepts::detail::view_distance_metric<DistanceMetric, Ndim>) {
      return metric(view, i, j);
    } else {
      return metric(view[i], view[j]);
    }
  }

  template <std::size_t Ndim,
            std::floating_point TData,
            clue::concepts::distance_metric<Ndim> DistanceMetric>
  inline auto silhouette(const clue::AssociationMapHost& clusters,
                         const clue::PointsHost<Ndim, TData>& points,
                         int point,
                         const DistanceMetric& metric) {
    auto a = TData{0};
    std::vector<TData> b_values;
    b_values.reserve(clusters.size() - 1);

    a += std::accumulate(clusters.lower_bound(points[point].cluster_index()),
                         clusters.upper_bound(points[point].cluster_index()),
                         TData{0},
                         [&](TData acc, int other_point) {
                           if (other_point == point)
                             return acc;
                           return acc + distance(points.view(), metric, point, other_point);
                         });
    a /= static_cast<TData>(clusters.count(points[point].cluster_index()) - 1);
    for (auto cluster_idx = 0; cluster_idx < static_cast<int32_t>(clusters.size()); ++cluster_idx) {
      if (cluster_idx == points[point].cluster_index())
        continue;
      auto b = TData{0};
      b += std::accumulate(clusters.lower_bound(cluster_idx),
                           clusters.upper_bound(cluster_idx),
                           TData{0},
                           [&](TData acc, int other_point) {
                             return acc + distance(points.view(), metric, point, other_point);
                           });
      b /= static_cast<TData>(clusters.count(cluster_idx));
      b_values.push_back(b);
    }
    const auto b = std::reduce(b_values.begin(),
                               b_values.end(),
                               std::numeric_limits<TData>::max(),
                               [](auto acc, auto val) { return std::min(acc, val); });

    return (b - a) / std::max(a, b);
  }

}  // namespace detail

template <std::size_t Ndim,
          std::floating_point InputType,
          clue::concepts::distance_metric<Ndim> DistanceMetric>
  requires std::same_as<typename DistanceMetric::value_type, InputType>
inline auto silhouette(const clue::PointsHost<Ndim, InputType>& points,
                       int point,
                       const DistanceMetric& metric) {
  const auto clusters = clue::get_clusters(points);

  return detail::silhouette<Ndim>(clusters, points, point, metric);
}

template <std::size_t Ndim,
          std::floating_point InputType,
          clue::concepts::distance_metric<Ndim> DistanceMetric>
  requires std::same_as<typename DistanceMetric::value_type, InputType>
inline auto silhouette(const clue::PointsHost<Ndim, InputType>& points,
                       const DistanceMetric& metric) {
  const auto clusters = clue::get_clusters(points);
  std::vector<InputType> scores;
  auto valid_point = [&](int point) -> bool { return points[point].cluster_index() != -1; };
  auto valid_cluster = [&](int point) -> bool {
    return clusters.count(points[point].cluster_index()) >= 2;
  };
  auto compute_silhouette = [&](std::size_t point) -> InputType {
    return detail::silhouette(clusters, points, point, metric);
  };
  std::ranges::copy(std::views::iota(0) | std::views::take(points.size()) |
                        std::views::filter(valid_point) | std::views::filter(valid_cluster) |
                        std::views::transform(compute_silhouette),
                    std::back_inserter(scores));

  return std::reduce(scores.begin(), scores.end(), InputType{0}) /
         static_cast<InputType>(scores.size());
}

CLUE_NAMESPACE_END
