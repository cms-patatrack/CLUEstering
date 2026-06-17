
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
      WeightedEuclidean,
      PeriodicEuclidean,
      Manhattan,
      WeightedChebyshev
    };

    Tag tag = Tag::WeightedEuclidean;
    std::vector<TData> params;  // empty means unit weights for weighted metrics
  };

  template <std::size_t Ndim, std::floating_point TData, typename Callable>
  void apply_metric(const MetricDescriptor<TData>& desc, Callable&& callable) {
    using Tag = typename MetricDescriptor<TData>::Tag;

    switch (desc.tag) {
      case Tag::WeightedEuclidean: {
        std::array<TData, Ndim> weights;
        if (desc.params.empty()) {
          weights.fill(TData{1});
        } else {
          if (desc.params.size() != Ndim) {
            throw std::invalid_argument("EuclideanMetric requires exactly " + std::to_string(Ndim) +
                                        " weight(s), got " + std::to_string(desc.params.size()));
          }
          std::copy_n(desc.params.begin(), Ndim, weights.begin());
        }
        std::forward<Callable>(callable)(clue::WeightedEuclideanMetric<Ndim, TData>{weights});
        return;
      }

      case Tag::Manhattan: {
        std::forward<Callable>(callable)(clue::ManhattanMetric<Ndim, TData>{});
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
        std::array<TData, Ndim> weights;
        if (desc.params.empty()) {
          weights.fill(TData{1});
        } else {
          if (desc.params.size() != Ndim) {
            throw std::invalid_argument("ChebyshevMetric requires exactly " + std::to_string(Ndim) +
                                        " weight(s), got " + std::to_string(desc.params.size()));
          }
          std::copy_n(desc.params.begin(), Ndim, weights.begin());
        }
        std::forward<Callable>(callable)(clue::WeightedChebyshevMetric<Ndim, TData>{weights});
        return;
      }
    }
  }

}  // namespace clue::internal
