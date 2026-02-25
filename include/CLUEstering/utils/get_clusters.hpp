/// @file get_clusters.hpp
/// @brief Provides functions for building cluster-to-points association maps, both on host and device.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <concepts>
#include <cstddef>

namespace clue {

  /// @brief Construct a map associating clusters to points
  /// This overload works on host points and returns a map allocated on the host.
  ///
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TData The data type for the point coordinates and weights
  /// @param points The points for which to get the clusters
  /// @return An AssociationMap where each key is a cluster index and the associated values
  /// are the indices of the points belonging to that cluster
  template <std::size_t Ndim, std::floating_point TData = float>
  inline auto get_clusters(const PointsHost<Ndim, TData>& points);

  /// @brief Construct a map associating clusters to points
  /// This overload works on device points and returns a map allocated on the device.
  ///
  /// @tparam TQueue The type of queue to use for device computations
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TData The data type for the point coordinates and weights
  /// @param points The points for which to get the clusters
  /// @return An AssociationMap where each key is a cluster index and the associated values
  /// are the indices of the points belonging to that cluster
  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TData = float>
  inline auto get_clusters(TQueue& queue, const PointsDevice<Ndim, TData>& points);

}  // namespace clue

#include "CLUEstering/utils/detail/get_clusters.hpp"
