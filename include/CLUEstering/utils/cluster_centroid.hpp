/// @file cluster_centroid.hpp
/// @brief Provides functions for computing the centroids of clusters.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include <array>
#include <cstddef>
#include <concepts>
#include <vector>

namespace clue {

  /// @brief Type alias for a centroid in Ndim dimensions
  ///
  /// @tparam Ndim The number of dimensions of the centroids
  /// @tparam TData The data type for the centroid coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  using Centroid = std::array<TData, Ndim>;

  /// @brief Type alias for a collection of centroids in Ndim dimensions
  ///
  /// @tparam Ndim The number of dimensions of the Centroids
  /// @tparam TData The data type for the centroid coordinates
  template <std::size_t Ndim, std::floating_point TData = float>
  using Centroids = std::vector<Centroid<Ndim, TData>>;

  /// @brief Compute the centroid of a specific cluster from the given Points
  ///
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TData The data type for the point coordinates
  /// @param points The PointsHost object containing the points
  /// @param cluster_id The ID of the cluster for which to compute the centroid
  /// @return The centroid of the specified cluster as an array of floating points
  template <std::size_t Ndim, std::floating_point TData = float>
  inline Centroid<Ndim, TData> cluster_centroid(const clue::PointsHost<Ndim, TData>& points,
                                                std::size_t cluster_id);

  /// @brief Compute the centroids of all clusters from the given Points
  ///
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TData The data type for the point coordinates
  /// @param points The PointsHost object containing the points
  /// @return A vector of centroids, one for each cluster
  template <std::size_t Ndim, std::floating_point TData = float>
  inline Centroids<Ndim, TData> cluster_centroids(const clue::PointsHost<Ndim, TData>& points);

}  // namespace clue

#include "CLUEstering/utils/detail/cluster_centroid.hpp"
