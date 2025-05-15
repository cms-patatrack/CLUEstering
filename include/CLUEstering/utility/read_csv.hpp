
#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// TODO: implement overload that takes pre-allocated Points
template <typename T, size_t NDim>
inline std::vector<T> read_csv(const std::string& file_path) {
  std::fstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }
  auto n_points =
      std::count(
          std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') -
      1;
  std::vector<T> coords((NDim + 1) * n_points);

  file = std::fstream(file_path);
  // discard the header
  std::string buffer;
  getline(file, buffer);
  auto point_id = 0;
  while (getline(file, buffer)) {
    std::stringstream buffer_stream(buffer);
    std::string value;

    for (size_t dim = 0; dim <= NDim; ++dim) {
      getline(buffer_stream, value, ',');
      coords[point_id + dim * n_points] = static_cast<T>(std::stod(value));
    }
    ++point_id;
  }

  file.close();

  return coords;
}

template <size_t NDim>
inline std::vector<int> read_output(const std::string& file_path) {
  std::fstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }
  auto n_points =
      std::count(
          std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') -
      1;
  std::vector<int> results(2 * n_points);

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
    }
    getline(buffer_stream, value, ',');  // discard the weight value
    getline(buffer_stream, value, ',');
    results[point_id] = std::stoi(value);
    getline(buffer_stream, value);
    results[point_id + n_points] = std::stoi(value);

    ++point_id;
  }

  file.close();

  return results;
}
