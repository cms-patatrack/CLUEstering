
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include <array>
#include <vector>

namespace clue {

  template <uint8_t Ndim>
  using Centroid = std::array<float, Ndim>;

  template <uint8_t Ndim>
  using Centroids = std::vector<Centroid>;

  template <uint8_t Ndim>
  inline Centroid<Ndim> cluster_centroid(const clue::PointsHost<Ndim>& points, std::size_t cluster_id);

}

#include "CLUEstering/utils/detail/cluster_centroid.hpp"
