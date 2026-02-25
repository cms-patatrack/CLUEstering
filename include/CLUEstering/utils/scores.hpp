/// @file scores.hpp
/// @brief Provides fucntions for computing scores validating the quality of the clustering.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include <concepts>

namespace clue {

  /// @brief Compute the silhouette score for a specific point in the dataset.
  ///
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TData The data type for the point coordinates and weights, which must be a floating-point type.
  /// By default, it is set to `float`.
  /// @param points The dataset containing the points
  /// @param point The index of the point for which to compute the silhouette score
  /// @return The silhouette score of the specified point
  /// @note This function currently only works for points with non-periodic coordinates.
  template <std::size_t Ndim, std::floating_point TData = float>
  auto silhouette(const clue::PointsHost<Ndim, TData>& points, std::size_t point);

  /// @brief Compute the average silhouette score for the entire dataset.
  ///
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TData The data type for the point coordinates and weights, which must be a floating-point type.
  /// By default, it is set to `float`.
  /// @param points The dataset containing the points
  /// @return The average silhouette score of the dataset
  /// @note This function currently only works for points with non-periodic coordinates.
  template <std::size_t Ndim, std::floating_point TData = float>
  auto silhouette(const clue::PointsHost<Ndim, TData>& points);

}  // namespace clue

#include "CLUEstering/utils/detail/scores.hpp"
