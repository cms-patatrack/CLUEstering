/// @file to_dot.hpp
/// @brief Provides a function to export clustering results in DOT format
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <concepts>
#include <cstddef>
#include <string>

namespace clue {

  /// @brief Export clustering results to a DOT format file
  ///
  /// @tparam TQueue The type of queue used for device memory copies
  /// @tparam Ndim The number of dimensions of the points
  /// @tparam TData The data type for the point coordinates and weights
  /// @tparam TDev The device type on which the points are allocated
  /// @param queue The queue used for device memory copies
  /// @param points The clustered device points to export
  /// @param file_path The path to the output DOT file
  /// @throws std::runtime_error if the points have not been clustered or the file cannot be opened
  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData = float,
            concepts::device TDev = clue::Device>
  inline void to_dot(TQueue& queue,
                     const PointsDevice<Ndim, TData, TDev>& points,
                     const std::string& file_path = "clusters.dot");

}  // namespace clue

#include "CLUEstering/utils/detail/to_dot.hpp"
