/// @file DistanceParameter.hpp
/// @brief Definition of the DistanceParameter class, which encapsulates distance parameters for clustering algorithms.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <alpaka/alpaka.hpp>

namespace clue {

  /// @brief A class that encapsulates distance parameters for the clustering algorithm.
  /// It can represent either a uniform distance across all dimensions or
  /// dimension-specific distances.
  template <std::size_t Ndim>
  class DistanceParameter {
  private:
    std::array<float, Ndim> m_parameters;

  public:
    /// @brief Construct a DistanceParameter with the same threshold for all dimensions.
    /// @param radius The distance threshold to be applied uniformly across all dimensions.
    /// @note This constructor initializes all dimensions with the same distance value.
    constexpr DistanceParameter(float radius) : m_parameters{} {
      std::ranges::fill(m_parameters, radius);
    }
    /// @brief Construct a DistanceParameter with dimension-specific distances.
    /// @param distances An array containing the distance thresholds for each dimension.
    /// @note This constructor allows for specifying different distance values for each dimension.
    constexpr DistanceParameter(std::array<float, Ndim> distances)
        : m_parameters{std::move(distances)} {}
    /// @brief Construct a DistanceParameter with dimension-specific distances.
    /// @tparam TDistances A parameter pack representing the distance thresholds for each dimension.
    /// @param distances The distance thresholds for each dimension.
    /// @note This constructor allows for specifying different distance values for each dimension.
    template <std::floating_point... TDistances>
      requires(sizeof...(TDistances) == Ndim)
    constexpr DistanceParameter(TDistances... distances) : m_parameters{distances...} {}

    /// @brief Access the distance parameter for a specific dimension.
    /// @param dim The dimension index for which to retrieve the distance parameter.
    /// @return A reference to the distance parameter for the specified dimension.
    constexpr const auto& operator[](std::size_t dim) const { return m_parameters[dim]; }
    /// @brief Access the distance parameter for a specific dimension.
    /// @param dim The dimension index for which to retrieve the distance parameter.
    /// @return A reference to the distance parameter for the specified dimension.
    constexpr auto& operator[](std::size_t dim) { return m_parameters[dim]; }

    constexpr bool operator<(float value) {
      return std::ranges::any_of(m_parameters, [=](auto x) -> bool { return x < value; });
    }
    constexpr bool operator<=(float value) {
      return std::ranges::any_of(m_parameters, [=](auto x) -> bool { return x <= value; });
    }

  private:
    // TODO: needed to avoid failed template substitution
    // to be removed when changing type of Ndim
    template <std::size_t N1, std::size_t N2>
      requires(N1 == N2)
    ALPAKA_FN_ACC friend bool operator<=(const std::array<float, N1>&,
                                         const DistanceParameter<N2>&);
    // TODO: needed to avoid failed template substitution
    // to be removed when changing type of Ndim
    template <std::size_t N1, std::size_t N2>
      requires(N1 == N2)
    ALPAKA_FN_ACC friend bool operator>(const std::array<float, N1>&, const DistanceParameter<N2>&);
  };

  template <std::size_t N1, std::size_t N2>
    requires(N1 == N2)
  ALPAKA_FN_ACC bool operator<=(std::array<float, N1>& lhs, const DistanceParameter<N2>& rhs) {
    return [&]<std::size_t... Ids>(std::index_sequence<Ids...>) -> bool {
      return ((lhs[Ids] <= rhs[Ids]) && ...);
    }(std::make_index_sequence<N1>{});
  }

  template <std::size_t N1, std::size_t N2>
    requires(N1 == N2)
  ALPAKA_FN_ACC bool operator>(std::array<float, N1>& lhs, const DistanceParameter<N2>& rhs) {
    return [&]<std::size_t... Ids>(std::index_sequence<Ids...>) -> bool {
      return ((lhs[Ids] > rhs[Ids]) && ...);
    }(std::make_index_sequence<N1>{});
  }

}  // namespace clue
