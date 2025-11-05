/// @file scores.hpp
/// @brief Provides fucntions for computing scores validating the quality of the clustering.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"

namespace clue {

  /// @brief Compute the silhouette score for a specific point in the dataset.
  ///
  /// @tparam Ndim The number of dimensions of the points
  /// @param points The dataset containing the points
  /// @param point The index of the point for which to compute the silhouette score
  /// @return The silhouette score of the specified point
  /// @note This function currently only works for points with non-periodic coordinates.
  template <std::size_t Ndim>
  auto silhouette(const clue::PointsHost<Ndim>& points, std::size_t point);

  /// @brief Compute the average silhouette score for the entire dataset.
  ///
  /// @tparam Ndim The number of dimensions of the points
  /// @param points The dataset containing the points
  /// @return The average silhouette score of the dataset
  /// @note This function currently only works for points with non-periodic coordinates.
  template <std::size_t Ndim>
  auto silhouette(const clue::PointsHost<Ndim>& points);

}  // namespace clue

#include "CLUEstering/utils/detail/scores.hpp"
