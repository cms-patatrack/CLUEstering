/// @file DistanceMetrics.hpp
/// @author Simone Balducci
/// @brief Implementations of common distance metrics

#pragma once

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/internal/meta/accumulate.hpp"
#include "CLUEstering/internal/meta/maximum.hpp"
#include "CLUEstering/internal/math/math.hpp"
#include <alpaka/alpaka.hpp>
#include <array>
#include <concepts>
#include <cstddef>

namespace clue {

  namespace concepts {

    namespace detail {

      /// Metric callable as metric(Point, Point) -> value_type
      template <typename TMetric, std::size_t Ndim>
      concept point_distance_metric =
          requires(TMetric&& metric, std::array<typename TMetric::value_type, Ndim + 1> arr) {
            { metric(arr, arr) } -> std::same_as<typename TMetric::value_type>;
          };

      /// Metric callable as metric(PointsView, i, j) -> value_type (used for per-point sigma)
      template <typename TMetric, std::size_t Ndim>
      concept view_distance_metric = requires(
          TMetric&& metric, PointsView<Ndim, typename TMetric::value_type> view, std::size_t i) {
        { metric(view, i, i) } -> std::same_as<typename TMetric::value_type>;
      };

    }  // namespace detail

    /// @brief Concept for distance metrics accepted by the clusterer
    ///
    /// Satisfied by either a point-wise metric (taking two coordinate arrays) or a
    /// view-wise metric (taking a PointsView and two point indices).
    template <typename TMetric, std::size_t Ndim>
    concept distance_metric =
        detail::point_distance_metric<TMetric, Ndim> || detail::view_distance_metric<TMetric, Ndim>;

  }  // namespace concepts

  template <std::size_t Ndim, std::floating_point TData>
  using Point = std::array<TData, Ndim + 1>;

  /// @brief Mahalanobis distance metric with per-point coordinate uncertainties
  ///
  /// Computes a normalised distance between two points using the per-point sigma
  /// values stored in the PointsView. For each dimension the contribution is
  /// `(x_i - x_j)^2 / (sigma_i^2 + sigma_j^2)`, which is the correct combination
  /// for two independent Gaussian uncertainties.
  ///
  /// @note Requires that sigma values have been set for every dimension via
  ///       `set_sigma` or `set_sigmas` before clustering.
  ///
  /// @tparam Ndim Number of dimensions
  /// @tparam TData Floating-point type for coordinates and sigma values
  template <std::size_t Ndim, std::floating_point TData = float>
  class MahalanobisMetric {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

    ALPAKA_FN_HOST_ACC constexpr MahalanobisMetric() = default;

    /// @brief Compute the Mahalanobis distance between points i and j
    ///
    /// @param points The PointsView holding coordinates and sigma arrays
    /// @param i Index of the first point
    /// @param j Index of the second point
    /// @return Mahalanobis distance between the two points
    ALPAKA_FN_HOST_ACC constexpr inline auto operator()(PointsView<Ndim, TData> points,
                                                        std::size_t i,
                                                        std::size_t j) const {
      const auto d_squared = meta::accumulate<Ndim>([&]<std::size_t Dim>() {
        const auto diff = points[i][Dim] - points[j][Dim];
        const auto sigma_i = points.sigma(Dim)[i];
        const auto sigma_j = points.sigma(Dim)[j];
        return diff * diff / (sigma_i * sigma_i + sigma_j * sigma_j);
      });
      return math::sqrt(d_squared);
    }
  };

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
    ALPAKA_FN_HOST_ACC constexpr EuclideanMetric() = default;

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
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclideanMetric(const std::array<value_type, Ndim>& weights)
        : m_weights{weights} {}
    /// @brief Move constructor euclidian metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedEuclideanMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclideanMetric(std::array<value_type, Ndim>&& weights)
        : m_weights{std::move(weights)} {}
    /// @brief Constructor euclidian metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedEuclideanMetric object
    template <std::floating_point... TValues>
      requires(sizeof...(TValues) == Ndim)
    ALPAKA_FN_HOST_ACC constexpr WeightedEuclideanMetric(TValues... weights)
        : m_weights{weights...} {}

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
    /// @brief Constructor periodic euclidian metric with periods
    ///
    /// @param periods Periods for each dimension
    /// If a coordinate is not periodic, the corresponding period should be set to 0.f
    /// @return PeriodicEuclideanMetric object
    template <std::floating_point... TValues>
      requires(sizeof...(TValues) == Ndim)
    ALPAKA_FN_HOST_ACC constexpr PeriodicEuclideanMetric(TValues... periods)
        : m_periods{periods...} {}

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
    ALPAKA_FN_HOST_ACC constexpr ManhattanMetric() = default;

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
    ALPAKA_FN_HOST_ACC constexpr ChebyshevMetric() = default;

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
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(const std::array<value_type, Ndim>& weights)
        : m_weights{weights} {}
    /// @brief Move constructor weighted chebyshev metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedChebyshevMetric object
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(std::array<value_type, Ndim>&& weights)
        : m_weights{std::move(weights)} {}
    /// @brief Constructor weighted chebyshev metric with weights
    ///
    /// @param weights Weights for each dimension
    /// @return WeightedChebyshevMetric object
    template <std::floating_point... TValues>
      requires(sizeof...(TValues) == Ndim)
    ALPAKA_FN_HOST_ACC constexpr WeightedChebyshevMetric(TValues... weights)
        : m_weights{weights...} {}

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

    /// @brief Alias for Mahalanobis distance metric
    ///
    /// 	@tparam Ndim Number of dimensions
    /// 	@tparam TData Point coordinates and weights data type
    template <std::size_t Ndim, std::floating_point TData = float>
    using Mahalanobis = clue::MahalanobisMetric<Ndim, TData>;

  }  // namespace metrics

}  // namespace clue
