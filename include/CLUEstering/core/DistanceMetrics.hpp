
#pragma once

#include "CLUEstering/internal/meta/accumulate.hpp"
#include "CLUEstering/core/internal/MetricInterface.hpp"
#include <alpaka/alpaka.hpp>
#include <array>
#include <cstddef>

namespace clue {

  template <std::size_t Ndim>
  class EuclidianMetric : public internal::MetricInterface<EuclidianMetric<Ndim>, Ndim> {
  public:
    ALPAKA_FN_HOST_ACC constexpr EuclidianMetric() = default;

  private:
    ALPAKA_FN_HOST_ACC constexpr auto distance(const std::array<float, Ndim>& lhs,
                                               const std::array<float, Ndim>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]); });
      return math::sqrt(distance2);
    }

    friend class internal::MetricInterface<EuclidianMetric<Ndim>, Ndim>;
  };

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
    ALPAKA_FN_HOST_ACC constexpr auto distance(const std::array<float, Ndim>& lhs,
                                               const std::array<float, Ndim>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>([&]<std::size_t Dim>() {
        return m_weights[Dim] * (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]);
      });
      return math::sqrt(distance2);
    }

    friend class internal::MetricInterface<WeightedEuclidianMetric<Ndim>, Ndim>;
  };

  template <std::size_t Ndim>
  class ManhattanMetric : public internal::MetricInterface<ManhattanMetric<Ndim>, Ndim> {
  public:
    ALPAKA_FN_HOST_ACC constexpr ManhattanMetric() = default;

  private:
    ALPAKA_FN_HOST_ACC constexpr auto distance(const std::array<float, Ndim>& lhs,
                                               const std::array<float, Ndim>& rhs) const {
      return meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return math::abs(lhs[Dim] - rhs[Dim]); });
    }

    friend class internal::MetricInterface<ManhattanMetric<Ndim>, Ndim>;
  };

  namespace metrics {

    template <std::size_t Ndim>
    using Euclidian = clue::EuclidianMetric<Ndim>;

    template <std::size_t Ndim>
    using WeightedEuclidian = clue::WeightedEuclidianMetric<Ndim>;

    template <std::size_t Ndim>
    using Manhattan = clue::ManhattanMetric<Ndim>;

  }  // namespace metrics

}  // namespace clue
