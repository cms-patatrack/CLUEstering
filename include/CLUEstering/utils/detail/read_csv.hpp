
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

namespace clue {

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
        points.coords(dim)[point_id] = std::stof(value);
      }
      getline(buffer_stream, value);
      points.weights()[point_id] = std::stof(value);
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
    clue::PointsHost<NDim> points(queue, n_points);
    points.mark_clustered();
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
        view.m_coords[dim][point_id] = std::stof(value);
      }
      getline(buffer_stream, value, ',');
      view.m_weight[point_id] = std::stof(value);
      getline(buffer_stream, value, ',');
      view.m_cluster_index[point_id] = std::stoi(value);

      ++point_id;
    }
    file.close();

    return points;
  }

}  // namespace clue
