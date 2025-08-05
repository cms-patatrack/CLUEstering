
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/utils/read_csv.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace clue {
  template <size_t NDim, concepts::queue TQueue>
  inline clue::PointsHost<NDim> read_csv(TQueue& queue, const std::string& file_path) {
    std::fstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + file_path);
    }
    auto n_points =
        std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') -
        1;

    clue::PointsHost<NDim> points(queue, n_points);

    file = std::fstream(file_path);
    // discard the header
    std::string buffer;
    getline(file, buffer);
    auto point_id = 0;
    while (getline(file, buffer)) {
      std::stringstream buffer_stream(buffer);
      std::string value;

      for (size_t dim = 0; dim < NDim; ++dim) {
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

  template <size_t NDim, concepts::queue TQueue>
  inline clue::PointsHost<NDim> read_output(TQueue& queue, const std::string& file_path) {
    std::fstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + file_path);
    }
    auto n_points =
        std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') -
        1;
    clue::PointsHost<NDim> points(queue, n_points);

    file = std::fstream(file_path);
    // discard the header
    std::string buffer;
    getline(file, buffer);
    auto point_id = 0;
    while (getline(file, buffer)) {
      std::stringstream buffer_stream(buffer);
      std::string value;

      for (size_t dim = 0; dim < NDim; ++dim) {
        getline(buffer_stream, value, ',');
        points.coords(dim)[point_id] = std::stof(value);
      }
      getline(buffer_stream, value, ',');
      points.weights()[point_id] = std::stof(value);
      getline(buffer_stream, value, ',');
      points.clusterIndexes()[point_id] = std::stoi(value);
      getline(buffer_stream, value);
      points.isSeed()[point_id] = std::stoi(value);

      ++point_id;
    }
    file.close();

    return points;
  }
}  // namespace clue
