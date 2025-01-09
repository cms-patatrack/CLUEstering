
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
std::pair<std::vector<VecArray<T, NDim>>, std::vector<T>> read_csv(
    const std::string& file_path) {
  std::vector<VecArray<T, NDim>> coords;
  std::vector<T> weights;

  std::fstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }

  // discard the header
  std::string buffer;
  getline(file, buffer);
  while (getline(file, buffer)) {
    std::stringstream buffer_stream(buffer);
    std::string value;

    VecArray<T, NDim> point;
    for (size_t i{}; i < NDim; ++i) {
      getline(buffer_stream, value, ',');
      point.push_back_unsafe(static_cast<T>(std::stod(value)));
    }
    coords.push_back(point);
    getline(buffer_stream, value);
    weights.push_back(static_cast<T>(std::stod(value)));
  }

  file.close();

  return std::make_pair(coords, weights);
}

#endif
