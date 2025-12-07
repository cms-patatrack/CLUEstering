/// @file DistanceMetrics.hpp
/// @author Simone Balducci
/// @brief Implementations of common distance metrics

#pragma once

#include "CLUEstering/internal/meta/accumulate.hpp"
#include "CLUEstering/internal/meta/maximum.hpp"
#include "CLUEstering/internal/math/math.hpp"
#include <alpaka/alpaka.hpp>
#include <array>
#include <cstddef>

namespace clue {

  namespace concepts {

    template <typename TMetric, std::size_t Ndim>
    concept distance_metric = requires(TMetric&& metric) {
      {
        metric(std::array<float, Ndim + 1>{}, std::array<float, Ndim + 1>{})
      } -> std::same_as<float>;
    };

  }  // namespace concepts

  template <std::size_t Ndim>
  using Point = std::array<float, Ndim + 1>;

  /// @brief Euclidian distance metric
  //// This class implements the Euclidian distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class EuclidianMetric {
  public:
    /// @brief Default constructor
    ///
    /// @return EuclidianMetric object
    ALPAKA_FN_HOST_ACC constexpr EuclidianMetric() = default;

    /// @brief Compute the Euclidian distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Euclidian distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim>& lhs,
                                                        const Point<Ndim>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]); });
      return math::sqrt(distance2);
    }
  };

  /// @brief Weighted Euclidian distance metric
  /// This class implements the Weighted Euclidian distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class WeightedEuclidianMetric {
  private:
    std::array<float, Ndim> m_weights;

  public:
    /// @brief Constructor euclidian metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedEuclidianMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclidianMetric(const std::array<float, Ndim>& weights)
        : m_weights{weights} {}
    /// @brief Move constructor euclidian metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedEuclidianMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclidianMetric(std::array<float, Ndim>&& weights)
        : m_weights{std::move(weights)} {}

    /// @brief Compute the Weighted Euclidian distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Weighted Euclidian distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim>& lhs,
                                                        const Point<Ndim>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>([&]<std::size_t Dim>() {
        return m_weights[Dim] * (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]);
      });
      return math::sqrt(distance2);
    }
  };

  /// @brief Manhattan distance metric
  /// This class implements the Manhattan distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class ManhattanMetric {
  public:
    /// @brief Default constructor
    ALPAKA_FN_HOST_ACC constexpr ManhattanMetric() = default;

    /// @brief Compute the Manhattan distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Manhattan distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim>& lhs,
                                                        const Point<Ndim>& rhs) const {
      return meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return math::fabs(lhs[Dim] - rhs[Dim]); });
    }
  };

  /// @brief Chebyshev distance metric
  /// This class implements the Chebyshev distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class ChebyshevMetric {
  public:
    /// @brief Default constructor
    ALPAKA_FN_HOST_ACC constexpr ChebyshevMetric() = default;

    /// @brief Compute the Chebyshev distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Chebyshev distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim>& lhs,
                                                        const Point<Ndim>& rhs) const {
      return meta::maximum<Ndim>(
          [&]<std::size_t Dim>() { return math::fabs(lhs[Dim] - rhs[Dim]); });
    }
  };

  /// @brief Weighted Chebyshev distance metric
  /// This class implements the weighted Chebyshev distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  template <std::size_t Ndim>
  class WeightedChebyshevMetric {
  private:
    std::array<float, Ndim> m_weights;

  public:
    /// @brief Constructor weighted chebyshev metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedChebyshevMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(const std::array<float, Ndim>& weights)
        : m_weights{weights} {}
    /// @brief Move constructor weighted chebyshev metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedChebyshevMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(std::array<float, Ndim>&& weights)
        : m_weights{std::move(weights)} {}

    /// @brief Compute the Weighted Chebyshev distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Weighted Chebyshev distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim>& lhs,
                                                        const Point<Ndim>& rhs) const {
      return meta::maximum<Ndim>(
          [&]<std::size_t Dim>() { return m_weights[Dim] * math::fabs(lhs[Dim] - rhs[Dim]); });
    }
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

    /// @brief Alias for Chebyshev distance metric
    template <std::size_t Ndim>
    using Chebyshev = clue::ChebyshevMetric<Ndim>;

    /// @brief Alias for Weighted Chebyshev distance metric
    template <std::size_t Ndim>
    using WeightedChebyshev = clue::WeightedChebyshevMetric<Ndim>;

  }  // namespace metrics

}  // namespace clue
