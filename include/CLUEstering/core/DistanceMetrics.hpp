
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
      return meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]); });
    }

    friend class internal::MetricInterface<EuclidianMetric<Ndim>, Ndim>;
  };

  namespace metrics {

    template <std::size_t Ndim>
    using Euclidian = clue::EuclidianMetric<Ndim>;

  }  // namespace metrics

}  // namespace clue
