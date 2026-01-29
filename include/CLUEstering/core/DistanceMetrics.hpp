/// @file DistanceMetrics.hpp
/// @author Simone Balducci
/// @brief Implementations of common distance metrics

#pragma once

#include "CLUEstering/internal/meta/accumulate.hpp"
#include "CLUEstering/internal/meta/maximum.hpp"
#include "CLUEstering/internal/math/math.hpp"
#include <alpaka/alpaka.hpp>
#include <array>
#include <concepts>
#include <cstddef>

namespace clue {

  namespace concepts {

    template <typename TMetric, std::size_t Ndim>
    concept distance_metric = requires(TMetric&& metric) {
      {
        metric(std::array<typename TMetric::value_type, Ndim + 1>{},
               std::array<typename TMetric::value_type, Ndim + 1>{})
      } -> std::same_as<typename TMetric::value_type>;
    };

  }  // namespace concepts

  template <std::size_t Ndim, std::floating_point TData>
  using Point = std::array<TData, Ndim + 1>;

  /// @brief Euclidean distance metric
  //// This class implements the Euclidean distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  /// @tparam TData Data type for the coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  class EuclideanMetric {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;
    /// @brief Default constructor
    ///
    /// @return EuclideanMetric object
    ALPAKA_FN_HOST_ACC constexpr EuclideanMetric() {}

    /// @brief Compute the Euclidean distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Euclidean distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim, value_type>& lhs,
                                                        const Point<Ndim, value_type>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]); });
      return math::sqrt(distance2);
    }
  };

  /// @brief Weighted Euclidean distance metric
  /// This class implements the Weighted Euclidean distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  /// @tparam TData Data type for the coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  class WeightedEuclideanMetric {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

  private:
    std::array<value_type, Ndim> m_weights;

  public:
    /// @brief Constructor euclidian metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedEuclideanMetric object
    template <std::floating_point... TValues>
      requires(sizeof...(TValues) == Ndim)
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclideanMetric(TValues... weights)
        : m_weights{weights...} {}
    /// @brief Constructor euclidian metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedEuclideanMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclideanMetric(const std::array<value_type, Ndim>& weights)
        : m_weights{weights} {}
    /// @brief Move constructor euclidian metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedEuclideanMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclideanMetric(std::array<value_type, Ndim>&& weights)
        : m_weights{std::move(weights)} {}

    /// @brief Compute the Weighted Euclidean distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Weighted Euclidean distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim, value_type>& lhs,
                                                        const Point<Ndim, value_type>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>([&]<std::size_t Dim>() {
        return m_weights[Dim] * (lhs[Dim] - rhs[Dim]) * (lhs[Dim] - rhs[Dim]);
      });
      return math::sqrt(distance2);
    }
  };

  /// @brief Periodic Euclidean distance metric
  /// This class implements the Euclidean distance metric in Ndim dimensions for coordinate systems
  /// where one or more coordinates are defined in a periodic domain. The distance is then computed
  /// by treating the coordinates as euclidean..
  ///  This metric expects a period for each dimension, where a period equal to 0 means that the coordinate
  ///  is not periodic. The periodic coordinates are expected to be defined in the range [0, period).
  ///
  /// @tparam Ndim Number of dimensions
  /// @tparam TData Data type for the coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  class PeriodicEuclideanMetric {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

  private:
    std::array<value_type, Ndim> m_periods;

  public:
    /// @brief Constructor periodic euclidian metric with periods
    ///
    /// @param periods Periods for each dimension
    /// If a coordinate is not periodic, the corresponding period should be set to 0.f
    /// @return PeriodicEuclideanMetric object
    ALPAKA_FN_HOST_ACC constexpr PeriodicEuclideanMetric(const std::array<value_type, Ndim>& periods)
        : m_periods{periods} {}
    /// @brief Move constructor periodic euclidian metric with periods
    ///
    /// @param periods Periods for each dimension
    /// If a coordinate is not periodic, the corresponding period should be set to 0.f
    /// @return PeriodicEuclideanMetric object
    ALPAKA_FN_HOST_ACC constexpr PeriodicEuclideanMetric(std::array<value_type, Ndim>&& periods)
        : m_periods{std::move(periods)} {}

    /// @brief Compute the Periodic Euclidean distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Periodic Euclidean distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim, value_type>& lhs,
                                                        const Point<Ndim, value_type>& rhs) const {
      const auto distance2 = meta::accumulate<Ndim>([&]<std::size_t Dim>() {
        const auto diff = math::fabs(lhs[Dim] - rhs[Dim]);
        const auto periodic_diff = math::min(diff, m_periods[Dim] - diff);
        return periodic_diff * periodic_diff;
      });
      return math::sqrt(distance2);
    }
  };

  /// @brief Manhattan distance metric
  /// This class implements the Manhattan distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  /// @tparam TData Data type for the coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  class ManhattanMetric {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

    /// @brief Default constructor
    ALPAKA_FN_HOST_ACC constexpr ManhattanMetric() {}

    /// @brief Compute the Manhattan distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Manhattan distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim, value_type>& lhs,
                                                        const Point<Ndim, value_type>& rhs) const {
      return meta::accumulate<Ndim>(
          [&]<std::size_t Dim>() { return math::fabs(lhs[Dim] - rhs[Dim]); });
    }
  };

  /// @brief Chebyshev distance metric
  /// This class implements the Chebyshev distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  /// @tparam TData Data type for the coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  class ChebyshevMetric {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

    /// @brief Default constructor
    ALPAKA_FN_HOST_ACC constexpr ChebyshevMetric() {}

    /// @brief Compute the Chebyshev distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Chebyshev distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim, value_type>& lhs,
                                                        const Point<Ndim, value_type>& rhs) const {
      return meta::maximum<Ndim>(
          [&]<std::size_t Dim>() { return math::fabs(lhs[Dim] - rhs[Dim]); });
    }
  };

  /// @brief Weighted Chebyshev distance metric
  /// This class implements the weighted Chebyshev distance metric in Ndim dimensions.
  ///
  /// @tparam Ndim Number of dimensions
  /// @tparam TData Data type for the coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  class WeightedChebyshevMetric {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

  private:
    std::array<value_type, Ndim> m_weights;

  public:
    /// @brief Constructor weighted chebyshev metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedChebyshevMetric object
    template <std::floating_point... TValues>
      requires(sizeof...(TValues) == Ndim)
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(TValues... weights)
        : m_weights{weights...} {}
    /// @brief Constructor weighted chebyshev metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedChebyshevMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(const std::array<value_type, Ndim>& weights)
        : m_weights{weights} {}
    /// @brief Move constructor weighted chebyshev metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedChebyshevMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(std::array<value_type, Ndim>&& weights)
        : m_weights{std::move(weights)} {}

    /// @brief Compute the Weighted Chebyshev distance between two points
    ///
    /// @param lhs First point
    /// @param rhs Second point
    /// @return Weighted Chebyshev distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(const Point<Ndim, value_type>& lhs,
                                                        const Point<Ndim, value_type>& rhs) const {
      return meta::maximum<Ndim>(
          [&]<std::size_t Dim>() { return m_weights[Dim] * math::fabs(lhs[Dim] - rhs[Dim]); });
    }
  };

  namespace metrics {

    /// @brief Alias for Euclidean distance metric
    ///
    /// 	@tparam Ndim Number of dimensions
    /// 	@tparam TData Point coordinates and weights data type
    template <std::size_t Ndim, std::floating_point TData = float>
    using Euclidean = clue::EuclideanMetric<Ndim, TData>;

    /// @brief Alias for Weighted Euclidean distance metric
    ///
    /// 	@tparam Ndim Number of dimensions
    /// 	@tparam TData Point coordinates and weights data type
    template <std::size_t Ndim, std::floating_point TData = float>
    using WeightedEuclidean = clue::WeightedEuclideanMetric<Ndim, TData>;

    /// @brief Alias for Periodic Euclidean distance metric
    ///
    /// 	@tparam Ndim Number of dimensions
    /// 	@tparam TData Point coordinates and weights data type
    template <std::size_t Ndim, std::floating_point TData = float>
    using PeriodicEuclidean = clue::PeriodicEuclideanMetric<Ndim, TData>;

    /// @brief Alias for Manhattan distance metric
    ///
    /// 	@tparam Ndim Number of dimensions
    /// 	@tparam TData Point coordinates and weights data type
    template <std::size_t Ndim, std::floating_point TData = float>
    using Manhattan = clue::ManhattanMetric<Ndim, TData>;

    /// @brief Alias for Chebyshev distance metric
    ///
    /// 	@tparam Ndim Number of dimensions
    /// 	@tparam TData Point coordinates and weights data type
    template <std::size_t Ndim, std::floating_point TData = float>
    using Chebyshev = clue::ChebyshevMetric<Ndim, TData>;

    /// @brief Alias for Weighted Chebyshev distance metric
    ///
    /// 	@tparam Ndim Number of dimensions
    /// 	@tparam TData Point coordinates and weights data type
    template <std::size_t Ndim, std::floating_point TData = float>
    using WeightedChebyshev = clue::WeightedChebyshevMetric<Ndim, TData>;

  }  // namespace metrics

}  // namespace clue
