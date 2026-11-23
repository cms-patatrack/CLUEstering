
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/utils/read_csv.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>

CLUE_NAMESPACE_BEGIN

namespace detail {

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

  inline bool is_blank(const std::string& line) {
    return std::ranges::all_of(line, [](unsigned char c) { return std::isspace(c); });
  }

  inline std::size_t count_rows(std::ifstream& file, const std::string& file_path) {
    std::string line;
    std::getline(file, line);
    std::size_t n_rows = 0;
    while (std::getline(file, line)) {
      if (!is_blank(line)) {
        ++n_rows;
      }
    }
    if (n_rows == 0) {
      throw std::runtime_error("The file does not contain any point: " + file_path);
    }
    file.clear();
    file.seekg(0);
    std::getline(file, line);
    return n_rows;
  }

}  // namespace detail

template <std::size_t NDim, std::floating_point TData, concepts::queue TQueue>
inline clue::PointsHost<NDim, TData> read_csv(TQueue& queue, const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }
  const auto n_points = detail::count_rows(file, file_path);

  clue::PointsHost<NDim, TData> points(queue, static_cast<int32_t>(n_points));

  std::string buffer;
  auto point_id = 0;
  while (getline(file, buffer)) {
    if (detail::is_blank(buffer)) {
      continue;
    }
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
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }
  const auto n_points = detail::count_rows(file, file_path);
  clue::PointsHost<NDim, TData> points(queue, static_cast<int32_t>(n_points));
  internal::points_interface<std::remove_cvref_t<decltype(points)>>::mark_clustered(points);
  auto& view = points.view();

  std::string buffer;
  auto point_id = 0;
  while (getline(file, buffer)) {
    if (detail::is_blank(buffer)) {
      continue;
    }
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

CLUE_NAMESPACE_END
