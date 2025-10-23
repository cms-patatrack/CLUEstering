/// @file cluster_centroid.hpp
/// @brief Provides functions for computing the centroids of clusters.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include <array>
#include <vector>

namespace clue {

  /// @brief Type alias for a centroid in Ndim dimensions
  template <std::size_t Ndim>
  using Centroid = std::array<float, Ndim>;

  /// @brief Type alias for a collection of centroids in Ndim dimensions
  template <std::size_t Ndim>
  using Centroids = std::vector<Centroid<Ndim>>;

  template <std::size_t Ndim>
  inline Centroid<Ndim> cluster_centroid(const clue::PointsHost<Ndim>& points,
                                         std::size_t cluster_id);

  template <std::size_t Ndim>
  inline Centroids<Ndim> cluster_centroids(const clue::PointsHost<Ndim>& points);

}  // namespace clue

#include "CLUEstering/utils/detail/cluster_centroid.hpp"
