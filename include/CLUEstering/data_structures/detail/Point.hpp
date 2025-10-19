
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"

namespace clue {

  template <std::size_t Ndim>
  inline PointsHost<Ndim>::Point::Point(const std::array<float, Ndim>& coordinates,
                                        float weight,
                                        int cluster_index)
      : m_coordinates(coordinates), m_weight(weight), m_clusterIndex(cluster_index) {}

  template <std::size_t Ndim>
  inline float PointsHost<Ndim>::Point::operator[](size_t dim) const {
    return m_coordinates[dim];
  }

  template <std::size_t Ndim>
  inline float PointsHost<Ndim>::Point::weight() const {
    return m_weight;
  }
  template <std::size_t Ndim>
  inline float PointsHost<Ndim>::Point::cluster_index() const {
    return m_clusterIndex;
  }

}  // namespace clue
