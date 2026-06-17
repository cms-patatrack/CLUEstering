
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/utils/read_csv.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace clue {

  namespace detail {

    /// @brief Parses a string into a floating-point value of the requested type.
    /// @details Dispatches to the std::sto* overload matching T so that the full
    /// precision of T is preserved (std::stof alone would truncate to float).
    template <std::floating_point T>
    inline T parse_floating(const std::string& value) {
      if constexpr (std::is_same_v<T, float>) {
        return std::stof(value);
      } else if constexpr (std::is_same_v<T, double>) {
        return std::stod(value);
      } else {
        return std::stold(value);
      }
    }

  }  // namespace detail

  template <std::size_t NDim, std::floating_point TData, concepts::queue TQueue>
  inline clue::PointsHost<NDim, TData> read_csv(TQueue& queue, const std::string& file_path) {
    std::fstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + file_path);
    }
    auto n_points =
        std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') -
        1;

    clue::PointsHost<NDim, TData> points(queue, n_points);

    file = std::fstream(file_path);
    // discard the header
    std::string buffer;
    getline(file, buffer);
    auto point_id = 0;
    while (getline(file, buffer)) {
      std::stringstream buffer_stream(buffer);
      std::string value;

      for (auto dim = 0u; dim < NDim; ++dim) {
        getline(buffer_stream, value, ',');
        points.coords(dim)[point_id] = detail::parse_floating<TData>(value);
      }
      getline(buffer_stream, value);
      points.weights()[point_id] = detail::parse_floating<TData>(value);
      ++point_id;
    }
    file.close();

    return points;
  }

  template <std::size_t NDim, std::floating_point TData, concepts::queue TQueue>
  inline clue::PointsHost<NDim, TData> read_output(TQueue& queue, const std::string& file_path) {
    std::fstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + file_path);
    }
    auto n_points =
        std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') -
        1;
    clue::PointsHost<NDim, TData> points(queue, n_points);
    internal::points_interface<std::remove_cvref_t<decltype(points)>>::mark_clustered(points);
    auto& view = points.view();

    file = std::fstream(file_path);
    // discard the header
    std::string buffer;
    getline(file, buffer);
    auto point_id = 0;
    while (getline(file, buffer)) {
      std::stringstream buffer_stream(buffer);
      std::string value;

      for (auto dim = 0u; dim < NDim; ++dim) {
        getline(buffer_stream, value, ',');
        view.m_coords[dim][point_id] = detail::parse_floating<TData>(value);
      }
      getline(buffer_stream, value, ',');
      view.m_weight[point_id] = detail::parse_floating<TData>(value);
      getline(buffer_stream, value, ',');
      view.m_cluster_index[point_id] = std::stoi(value);

      ++point_id;
    }
    file.close();

    return points;
  }

}  // namespace clue
