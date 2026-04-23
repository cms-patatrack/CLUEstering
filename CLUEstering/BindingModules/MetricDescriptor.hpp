
#pragma once

#include "CLUEstering/core/DistanceMetrics.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clue::internal {

  template <std::floating_point TData>
  struct MetricDescriptor {
    enum class Tag : std::uint8_t {
      Euclidean,
      WeightedEuclidean,
      PeriodicEuclidean,
      Manhattan,
      Chebyshev,
      WeightedChebyshev
    };

    Tag tag = Tag::Euclidean;
    std::vector<TData> params;
  };

  template <std::size_t Ndim, std::floating_point TData, typename Callable>
  void apply_metric(const MetricDescriptor<TData>& desc, Callable&& callable) {
    using Tag = typename MetricDescriptor<TData>::Tag;

    switch (desc.tag) {
      case Tag::Euclidean: {
        std::forward<Callable>(callable)(clue::EuclideanMetric<Ndim, TData>{});
        return;
      }

      case Tag::Manhattan: {
        std::forward<Callable>(callable)(clue::ManhattanMetric<Ndim, TData>{});
        return;
      }

      case Tag::Chebyshev: {
        std::forward<Callable>(callable)(clue::ChebyshevMetric<Ndim, TData>{});
        return;
      }

      case Tag::WeightedEuclidean: {
        if (desc.params.size() != Ndim) {
          throw std::invalid_argument("WeightedEuclideanMetric requires exactly " +
                                      std::to_string(Ndim) + " weight(s), got " +
                                      std::to_string(desc.params.size()));
        }
        std::array<TData, Ndim> weights;
        std::copy_n(desc.params.begin(), Ndim, weights.begin());
        std::forward<Callable>(callable)(clue::WeightedEuclideanMetric<Ndim, TData>{weights});
        return;
      }

      case Tag::PeriodicEuclidean: {
        if (desc.params.size() != Ndim) {
          throw std::invalid_argument("PeriodicEuclideanMetric requires exactly " +
                                      std::to_string(Ndim) + " period(s), got " +
                                      std::to_string(desc.params.size()));
        }
        std::array<TData, Ndim> periods;
        std::copy_n(desc.params.begin(), Ndim, periods.begin());
        std::forward<Callable>(callable)(clue::PeriodicEuclideanMetric<Ndim, TData>{periods});
        return;
      }

      case Tag::WeightedChebyshev: {
        if (desc.params.size() != Ndim) {
          throw std::invalid_argument("WeightedChebyshevMetric requires exactly " +
                                      std::to_string(Ndim) + " weight(s), got " +
                                      std::to_string(desc.params.size()));
        }
        std::array<TData, Ndim> weights;
        std::copy_n(desc.params.begin(), Ndim, weights.begin());
        std::forward<Callable>(callable)(clue::WeightedChebyshevMetric<Ndim, TData>{weights});
        return;
      }
    }
  }

}  // namespace clue::internal
