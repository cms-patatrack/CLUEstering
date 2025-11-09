/// @file PointsConversion.hpp
/// @brief Provides utilities for copying data between host and device points
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>

namespace clue {

  /// @brief Copies the results of the clustering from the device points to
  /// the host points
  ///
  /// @tparam TQueue The type of the queue for the device operations
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TDev The type of device that the points are allocated on
  /// @param queue The queue used for the device operations
  /// @param h_points The points allocated on the host, where the clustering results will be saved
  /// @param d_points The points allocated on the device, where the clustering has been run
  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToHost(TQueue& queue,
                  PointsHost<Ndim>& h_points,
                  const PointsDevice<Ndim, TDev>& d_points);

  /// @brief Copies the results of the clustering from the device points to
  /// the host points
  ///
  /// @tparam TQueue The type of the queue for the device operations
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TDev The type of device that the points are allocated on
  /// @param queue The queue used for the device operations
  /// @param h_points The points allocated on the host, where the clustering results will be saved
  /// @param d_points The points allocated on the device, where the clustering has been run
  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  auto copyToHost(TQueue& queue, const PointsDevice<Ndim, TDev>& d_points);

  /// @brief Copies the coordinates and weights of the points from the host to the device
  ///
  /// @tparam TQueue The type of the queue for the device operations
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TDev The type of device that the points are allocated on
  /// @param queue The queue used for the device operations
  /// @param d_points The empty points allocated on the device
  /// @param h_points The points allocated on the host, containing the points' coordinates
  /// and weights
  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToDevice(TQueue& queue,
                    PointsDevice<Ndim, TDev>& d_points,
                    const PointsHost<Ndim>& h_points);

  /// @brief Copies the coordinates and weights of the points from the host to the device
  ///
  /// @tparam TQueue The type of the queue for the device operations
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TDev The type of device that the points are allocated on
  /// @param queue The queue used for the device operations
  /// @param d_points The empty points allocated on the device
  /// @param h_points The points allocated on the host, containing the points' coordinates
  /// and weights
  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  auto copyToDevice(TQueue& queue, const PointsHost<Ndim>& h_points);

}  // namespace clue
