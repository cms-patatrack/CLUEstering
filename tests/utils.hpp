#pragma once

#include <concepts>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

  namespace detail {

    template <std::floating_point T>
    inline std::size_t parse_csv(const std::string& path, std::vector<std::vector<T>>& points) {
      std::ifstream file(path);
      if (!file.is_open()) {
        throw std::runtime_error("test: cannot open file: " + path);
      }

      points.clear();
      std::size_t n_dims = 0;
      std::string line;
      std::size_t line_no = 0;

      while (std::getline(file, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#' ||
            (line.size() >= 2 && line[0] == '/' && line[1] == '/')) {
          continue;
        }

        auto first = line.find_first_not_of(" \t\r\n");
        if (first != std::string::npos && std::isalpha(static_cast<unsigned char>(line[first]))) {
          continue;
        }

        std::vector<T> row;
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
          // Trim whitespace
          auto start = token.find_first_not_of(" \t\r\n");
          auto end = token.find_last_not_of(" \t\r\n");
          if (start == std::string::npos)
            continue;
          row.push_back(std::stof(token.substr(start, end - start + 1)));
        }

        if (row.size() < 2) {
          throw std::runtime_error("test: line " + std::to_string(line_no) +
                                   " has fewer than 2 columns (need at least 1 "
                                   "coordinate + weight)");
        }

        if (points.empty()) {
          n_dims = row.size() - 1;
        } else if (row.size() - 1 != n_dims) {
          throw std::runtime_error("test: inconsistent column count at line " +
                                   std::to_string(line_no));
        }

        points.push_back(std::move(row));
      }

      if (points.empty()) {
        throw std::runtime_error("test: file is empty or has no data rows: " + path);
      }

      return n_dims;
    }

  }  // namespace detail

  template <std::floating_point T>
  struct CombinedBuffers {
    std::vector<T> data;

    std::vector<int> labels;

    std::size_t n_points = 0;
    std::size_t n_dims = 0;
  };

  template <std::floating_point T>
  struct SeparateBuffers {
    std::vector<T> coords;

    std::vector<T> weights;

    std::vector<int> labels;

    std::size_t n_points = 0;
    std::size_t n_dims = 0;
  };

  template <std::floating_point T>
  struct PerDimBuffers {
    std::vector<std::vector<T>> dims;

    std::vector<T> weights;

    std::vector<int> labels;

    std::size_t n_points = 0;
    std::size_t n_dims = 0;
  };

  template <std::floating_point T>
  inline CombinedBuffers<T> read_csv_combined(const std::string& path) {
    std::vector<std::vector<T>> points;
    const std::size_t n_dims = detail::parse_csv(path, points);
    const std::size_t n_points = points.size();
    const std::size_t stride = n_dims + 1;

    CombinedBuffers<T> out;
    out.n_points = n_points;
    out.n_dims = n_dims;
    out.data.resize(n_points * stride);
    out.labels.assign(n_points, 0);

    for (auto i = 0u; i < n_points; ++i) {
      for (auto col = 0u; col < stride; ++col) {
        out.data[i * stride + col] = points[i][col];
      }
    }

    return out;
  }

  template <std::floating_point T>
  inline SeparateBuffers<T> read_csv_separate(const std::string& path) {
    std::vector<std::vector<T>> points;
    const std::size_t n_dims = detail::parse_csv(path, points);
    const std::size_t n_points = points.size();

    SeparateBuffers<T> out;
    out.n_points = n_points;
    out.n_dims = n_dims;
    out.coords.resize(n_points * n_dims);
    out.weights.resize(n_points);
    out.labels.assign(n_points, 0);

    for (auto i = 0u; i < n_points; ++i) {
      for (auto d = 0u; d < n_dims; ++d) {
        out.coords[i * n_dims + d] = points[i][d];
      }
      out.weights[i] = points[i][n_dims];
    }

    return out;
  }

  template <std::floating_point T>
  inline PerDimBuffers<T> read_csv_per_dim(const std::string& path) {
    std::vector<std::vector<T>> points;
    const std::size_t n_dims = detail::parse_csv(path, points);
    const std::size_t n_points = points.size();

    PerDimBuffers<T> out;
    out.n_points = n_points;
    out.n_dims = n_dims;
    out.dims.assign(n_dims, std::vector<T>(n_points));
    out.weights.resize(n_points);
    out.labels.assign(n_points, 0);

    for (auto i = 0u; i < n_points; ++i) {
      for (auto d = 0u; d < n_dims; ++d) {
        out.dims[d][i] = points[i][d];
      }
      out.weights[i] = points[i][n_dims];
    }

    return out;
  }

}  // namespace test
