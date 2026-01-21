
#pragma once

#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/utils/cluster_centroid.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

namespace clue {

  namespace detail {

    template <std::size_t Ndim>
    using Point = typename clue::PointsHost<Ndim>::Point;

    template <std::size_t Ndim>
    inline auto distance(const Point<Ndim>& lhs, const Point<Ndim>& rhs) {
      auto dist = 0.f;
      for (auto dim = 0u; dim < Ndim; ++dim) {
        dist += (lhs[dim] - rhs[dim]) * (lhs[dim] - rhs[dim]);
      }
      return std::sqrt(dist);
    }

    template <std::size_t Ndim>
    inline auto silhouette(const clue::host_associator& clusters,
                           const clue::PointsHost<Ndim>& points,
                           int point) {
      auto a = 0.f;
      std::vector<float> b_values;
      b_values.reserve(clusters.size() - 1);

      a +=
          std::accumulate(clusters.lower_bound(points[point].cluster_index()),
                          clusters.upper_bound(points[point].cluster_index()),
                          0.f,
                          [&](float acc, int other_point) {
                            if (other_point == point)
                              return acc;
                            return acc + detail::distance<Ndim>(points[point], points[other_point]);
                          });
      a /= static_cast<float>(clusters.count(points[point].cluster_index()) - 1);
      for (auto cluster_idx = 0; cluster_idx < static_cast<int32_t>(clusters.size());
           ++cluster_idx) {
        if (cluster_idx == points[point].cluster_index())
          continue;
        auto b = 0.f;
        b += std::accumulate(clusters.lower_bound(cluster_idx),
                             clusters.upper_bound(cluster_idx),
                             0.f,
                             [&](float acc, int other_point) {
                               return acc +
                                      detail::distance<Ndim>(points[point], points[other_point]);
                             });
        b /= static_cast<float>(clusters.count(cluster_idx));
        b_values.push_back(b);
      }
      const auto b = std::reduce(b_values.begin(),
                                 b_values.end(),
                                 std::numeric_limits<float>::max(),
                                 [](auto acc, auto val) { return std::min(acc, val); });

      return (b - a) / std::max(a, b);
    }

  }  // namespace detail

  template <std::size_t Ndim>
  inline auto silhouette(const clue::PointsHost<Ndim>& points, int point) {
    const auto clusters = clue::get_clusters(points);

    return detail::silhouette<Ndim>(clusters, points, point);
  }

  template <std::size_t Ndim>
  inline auto silhouette(const clue::PointsHost<Ndim>& points) {
    const auto clusters = clue::get_clusters(points);
    std::vector<float> scores;
    auto valid_point = [&](int point) -> bool { return points[point].cluster_index() != -1; };
    auto valid_cluster = [&](int point) -> bool {
      return clusters.count(points[point].cluster_index()) >= 2;
    };
    auto compute_silhouette = [&](std::size_t point) -> float {
      return detail::silhouette(clusters, points, point);
    };
    std::ranges::copy(std::views::iota(0) | std::views::take(points.size()) |
                          std::views::filter(valid_point) | std::views::filter(valid_cluster) |
                          std::views::transform(compute_silhouette),
                      std::back_inserter(scores));

    return std::reduce(scores.begin(), scores.end(), 0.f) / static_cast<float>(scores.size());
  }

  template <std::size_t Ndim, concepts::distance_metric<Ndim> DistanceMetric>
  auto davies_bouldin(const clue::PointsHost<Ndim>& points, const DistanceMetric& metric) {
    auto cluster_centroids = clue::cluster_centroids(points);
    auto clusters = clue::get_clusters(points);

    std::vector<float> clusters_scatter(cluster_centroids.size(), 0.f);
    for (auto i = 0; i < points.size(); ++i) {
      auto cluster_id = points[i].cluster_index();
      if (cluster_id == -1)
        continue;
      clusters_scatter[cluster_id] += metric(points[i], cluster_centroids[cluster_id]);
    }
    for (auto i = 0; i < cluster_centroids.size(); ++i) {
      clusters_scatter[i] /= static_cast<float>(clusters.count(i));
    }
    std::vector<std::vector<float>> clusters_separation(
        cluster_centroids.size(), std::vector<float>(cluster_centroids.size(), 0.f));
    for (auto i = 0u; i < cluster_centroids.size(); ++i) {
      for (auto j = 0u; j < cluster_centroids.size(); ++j) {
        if (i == j)
          continue;
        clusters_separation[i][j] = metric(cluster_centroids[i], cluster_centroids[j]);
      }
    }

    std::vector<float> R_values(cluster_centroids.size(), 0.f);
    for (auto i = 0u; i < cluster_centroids.size(); ++i) {
      for (auto j = 0u; j < clusters_separation[i].size(); ++j) {
        if (i == j)
          continue;
        R_values[i] = std::max(
            R_values[i], (clusters_scatter[i] + clusters_scatter[j]) / clusters_separation[i][j]);
      }
    }

    return std::reduce(R_values.begin(), R_values.end(), 0.f) / static_cast<float>(R_values.size());
  }

}  // namespace clue
