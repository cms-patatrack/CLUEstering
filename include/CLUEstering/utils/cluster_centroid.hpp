
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include <array>
#include <vector>

namespace clue {

  template <std::size_t Ndim>
  using Centroid = std::array<float, Ndim>;

  template <std::size_t Ndim>
  using Centroids = std::vector<Centroid<Ndim>>;

  template <std::size_t Ndim>
  inline Centroid<Ndim> cluster_centroid(const clue::PointsHost<Ndim>& points,
                                         std::size_t cluster_id);

  template <std::size_t Ndim>
  inline Centroids<Ndim> cluster_centroids(const clue::PointsHost<Ndim>& points);

}  // namespace clue

#include "CLUEstering/utils/detail/cluster_centroid.hpp"
