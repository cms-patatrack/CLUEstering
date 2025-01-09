
#ifndef read_csv_hpp
#define read_csv_hpp

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "CLUEstering/DataFormats/alpaka/AlpakaVecArray.hpp"

using clue::VecArray;

template <typename T, size_t NDim>
std::vector<T> read_csv(const std::string& file_path) {
  std::fstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }
  auto n_points = std::count(
      std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');

  std::vector<float> coords((NDim + 1) * n_points);

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

#endif
