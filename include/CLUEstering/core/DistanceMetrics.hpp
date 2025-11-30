/// @file DistanceMetrics.hpp
/// @author Simone Balducci
/// @brief Implementation of common distance metrics

#pragma once

#include "CLUEstering/internal/meta/accumulate.hpp"
#include "CLUEstering/core/internal/MetricInterface.hpp"
#include "CLUEstering/internal/math/math.hpp"
#include <alpaka/alpaka.hpp>
#include <array>
#include <cstddef>

namespace clue {

  template <std::size_t Ndim>
  using Point = std::array<float, Ndim + 1>;

  /// @brief Euclidian distance metric
  //// This class implements the Euclidian distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class EuclidianMetric : public internal::MetricInterface<EuclidianMetric<Ndim>, Ndim> {
  public:
    ALPAKA_FN_HOST_ACC constexpr EuclidianMetric() = default;

  private:
    ALPAKA_FN_HOST_ACC constexpr auto distance(const Point<Ndim>& lhs,
                                               const Point<Ndim>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]); });
      return math::sqrt(distance2);
    }

    friend class internal::MetricInterface<EuclidianMetric<Ndim>, Ndim>;
  };

  /// @brief Weighted Euclidian distance metric
  /// This class implements the Weighted Euclidian distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class WeightedEuclidianMetric
      : public internal::MetricInterface<WeightedEuclidianMetric<Ndim>, Ndim> {
  private:
    std::array<float, Ndim> m_weights;

  public:
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclidianMetric(const std::array<float, Ndim>& weights)
        : m_weights{weights} {}
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclidianMetric(std::array<float, Ndim>&& weights)
        : m_weights{std::move(weights)} {}

  private:
    ALPAKA_FN_HOST_ACC constexpr auto distance(const Point<Ndim>& lhs,
                                               const Point<Ndim>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>([&]<std::size_t Dim>() {
        return m_weights[Dim] * (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]);
      });
      return math::sqrt(distance2);
    }

    friend class internal::MetricInterface<WeightedEuclidianMetric<Ndim>, Ndim>;
  };

  /// @brief Manhattan distance metric
  /// This class implements the Manhattan distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class ManhattanMetric : public internal::MetricInterface<ManhattanMetric<Ndim>, Ndim> {
  public:
    ALPAKA_FN_HOST_ACC constexpr ManhattanMetric() = default;

  private:
    ALPAKA_FN_HOST_ACC constexpr auto distance(const Point<Ndim>& lhs,
                                               const Point<Ndim>& rhs) const {
      return meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return math::fabs(lhs[Dim] - rhs[Dim]); });
    }

    friend class internal::MetricInterface<ManhattanMetric<Ndim>, Ndim>;
  };

  namespace metrics {

    /// @brief Alias for Euclidian distance metric
    template <std::size_t Ndim>
    using Euclidian = clue::EuclidianMetric<Ndim>;

    /// @brief Alias for Weighted Euclidian distance metric
    template <std::size_t Ndim>
    using WeightedEuclidian = clue::WeightedEuclidianMetric<Ndim>;

    /// @brief Alias for Manhattan distance metric
    template <std::size_t Ndim>
    using Manhattan = clue::ManhattanMetric<Ndim>;

  }  // namespace metrics

}  // namespace clue
