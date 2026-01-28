
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

namespace clue {

  namespace detail {

    template <std::size_t Ndim, std::floating_point TData>
    using Point = typename clue::PointsHost<Ndim, TData>::Point;

    template <std::size_t Ndim, std::floating_point TData>
    inline auto distance(const Point<Ndim, TData>& lhs, const Point<Ndim, TData>& rhs) {
      auto dist = 0.f;
      for (auto dim = 0u; dim < Ndim; ++dim) {
        dist += (lhs[dim] - rhs[dim]) * (lhs[dim] - rhs[dim]);
      }
      return std::sqrt(dist);
    }

    template <std::size_t Ndim, std::floating_point TData>
    inline auto silhouette(const clue::host_associator& clusters,
                           const clue::PointsHost<Ndim, TData>& points,
                           int point) {
      auto a = 0.f;
      std::vector<TData> b_values;
      b_values.reserve(clusters.size() - 1);

      a += std::accumulate(clusters.lower_bound(points[point].cluster_index()),
                           clusters.upper_bound(points[point].cluster_index()),
                           0.f,
                           [&](TData acc, int other_point) {
                             if (other_point == point)
                               return acc;
                             return acc + detail::distance<Ndim, TData>(points[point],
                                                                        points[other_point]);
                           });
      a /= static_cast<TData>(clusters.count(points[point].cluster_index()) - 1);
      for (auto cluster_idx = 0; cluster_idx < static_cast<int32_t>(clusters.size());
           ++cluster_idx) {
        if (cluster_idx == points[point].cluster_index())
          continue;
        auto b = 0.f;
        b += std::accumulate(clusters.lower_bound(cluster_idx),
                             clusters.upper_bound(cluster_idx),
                             0.f,
                             [&](TData acc, int other_point) {
                               return acc + detail::distance<Ndim, TData>(points[point],
                                                                          points[other_point]);
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

  template <std::size_t Ndim>
  inline auto silhouette(const clue::PointsHost<Ndim>& points, int point) {
    const auto clusters = clue::get_clusters(points);

    return detail::silhouette<Ndim>(clusters, points, point);
  }

  template <std::size_t Ndim, std::floating_point TData>
  inline auto silhouette(const clue::PointsHost<Ndim, TData>& points) {
    const auto clusters = clue::get_clusters(points);
    std::vector<TData> scores;
    auto valid_point = [&](int point) -> bool { return points[point].cluster_index() != -1; };
    auto valid_cluster = [&](int point) -> bool {
      return clusters.count(points[point].cluster_index()) >= 2;
    };
    auto compute_silhouette = [&](std::size_t point) -> TData {
      return detail::silhouette(clusters, points, point);
    };
    std::ranges::copy(std::views::iota(0) | std::views::take(points.size()) |
                          std::views::filter(valid_point) | std::views::filter(valid_cluster) |
                          std::views::transform(compute_silhouette),
                      std::back_inserter(scores));

    return std::reduce(scores.begin(), scores.end(), 0.f) / static_cast<TData>(scores.size());
  }

}  // namespace clue
