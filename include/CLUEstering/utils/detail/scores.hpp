
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include <algorithm>
#include <numeric>
#include <ranges>
#include <span>
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
  }  // namespace detail

  template <std::size_t Ndim>
  inline auto silhouette(const clue::PointsHost<Ndim>& points, std::size_t point) {
    auto a = 0.f;
    auto b = 0.f;

    const auto pi = points[point];
    for (auto j = 0; j < points.size(); ++j) {
      if (static_cast<std::size_t>(j) == point)
        continue;
      const auto pj = points[j];
      const auto same_cluster = pi.cluster_index() == pj.cluster_index();
      a += same_cluster ? detail::distance<Ndim>(pi, pj) : 0.f;
      b += !same_cluster ? detail::distance<Ndim>(pi, pj) : 0.f;
    }

    return (b - a) / std::max(a, b);
  }

  template <std::size_t Ndim>
  inline auto silhouette(const clue::PointsHost<Ndim>& points) {
    std::vector<float> scores(points.size());
    std::ranges::transform(std::views::iota(0) | std::views::take(points.size()),
                           scores.begin(),
                           [&](auto point) -> float { return silhouette(points, point); });
    return std::reduce(scores.begin(), scores.end(), 0.f) / static_cast<float>(points.size());
  }

}  // namespace clue
