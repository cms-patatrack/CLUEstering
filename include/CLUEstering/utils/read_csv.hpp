/// @file read_csv.hpp
/// @brief Provides functions to read points from a CSV file into a PointsHost object
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <string>

namespace clue {

  /// @brief Read points from a CSV file into a PointsHost object
  ///
  /// @tparam NDim The number of dimensions of the points
  /// @tparam TQueue The type of the queue to use for reading the file
  /// @param queue The queue to use for reading the file
  /// @param file_path The path to the CSV file to read
  /// @return A PointsHost object containing the points read from the file
  template <size_t NDim, concepts::queue TQueue>
  inline clue::PointsHost<NDim> read_csv(TQueue& queue, const std::string& file_path);

  /// @brief Read output points from a CSV file into a PointsHost object
  ///
  /// @tparam NDim The number of dimensions of the points
  /// @tparam TQueue The type of the queue to use for reading the file
  /// @param queue The queue to use for reading the file
  /// @param file_path The path to the CSV file to read
  /// @return A PointsHost object containing the output points read from the file
  template <size_t NDim, concepts::queue TQueue>
  inline clue::PointsHost<NDim> read_output(TQueue& queue, const std::string& file_path);

}  // namespace clue

#include "CLUEstering/utils/detail/read_csv.hpp"
