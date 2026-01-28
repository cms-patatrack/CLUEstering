
#pragma once

#include <cstddef>
#include "CLUEstering/data_structures/PointsHost.hpp"

namespace clue {

  template <std::size_t Ndim, std::floating_point TData>
  inline PointsHost<Ndim, TData>::Point::Point(const std::array<value_type, Ndim>& coordinates,
                                               value_type weight,
                                               int cluster_index)
      : m_coordinates(coordinates), m_weight(weight), m_clusterIndex(cluster_index) {}

  template <std::size_t Ndim, std::floating_point TData>
  inline auto PointsHost<Ndim, TData>::Point::operator[](size_t dim) const {
    return m_coordinates[dim];
  }

  template <std::size_t Ndim, std::floating_point TData>
  inline auto PointsHost<Ndim, TData>::Point::weight() const {
    return m_weight;
  }
  template <std::size_t Ndim, std::floating_point TData>
  inline auto PointsHost<Ndim, TData>::Point::cluster_index() const {
    return m_clusterIndex;
  }

}  // namespace clue
