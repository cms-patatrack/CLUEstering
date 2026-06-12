
#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/utils/to_dot.hpp"

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace clue {

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::device TDev>
  inline void to_dot(TQueue& queue,
                     const PointsDevice<Ndim, TData, TDev>& d_points,
                     const std::string& file_path) {
    if (!d_points.clustered()) {
      throw std::runtime_error("Points have not been clustered yet");
    }

    const auto n_points = d_points.size();
    const auto& dev = alpaka::getDev(queue);

    auto h_cluster_index = make_host_buffer<int32_t[]>(queue, n_points);
    auto h_nearest_higher = make_host_buffer<int32_t[]>(queue, n_points);
    auto h_is_seed = make_host_buffer<int32_t[]>(queue, n_points);

    alpaka::memcpy(queue,
                   make_host_view(h_cluster_index.data(), n_points),
                   make_device_view(dev, d_points.view().m_cluster_index, n_points));
    alpaka::memcpy(queue,
                   make_host_view(h_nearest_higher.data(), n_points),
                   make_device_view(dev, d_points.view().m_nearest_higher, n_points));
    alpaka::memcpy(queue,
                   make_host_view(h_is_seed.data(), n_points),
                   make_device_view(dev, d_points.view().m_is_seed, n_points));
    alpaka::wait(queue);

    std::ofstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + file_path);
    }

    const auto* cluster_index = h_cluster_index.data();
    const auto* nearest_higher = h_nearest_higher.data();
    const auto* is_seed = h_is_seed.data();

    const auto max_cluster = *std::max_element(cluster_index, cluster_index + n_points);

    file << "digraph G {\n";
    file << "  node [shape=circle];\n";

    // Outlier subgraph (cluster_index == -1)
    const bool has_outliers = std::any_of(
        cluster_index, cluster_index + n_points, [](int idx) -> bool { return idx == -1; });
    if (has_outliers) {
      file << "  subgraph cluster_outliers {\n";
      file << "    label=\"Outliers\";\n";
      file << "    node [style=dashed];\n";
      for (int ii = 0; ii < n_points; ++ii) {
        if (cluster_index[ii] == -1) {
          file << "    " << ii << ";\n";
        }
      }
      file << "  }\n";
    }

    // One subgraph per cluster
    for (int cluster_id = 0; cluster_id <= max_cluster; ++cluster_id) {
      const bool has_points =
          std::any_of(cluster_index, cluster_index + n_points, [cluster_id](int idx) -> bool {
            return idx == cluster_id;
          });
      if (!has_points) {
        continue;
      }

      file << "  subgraph cluster_" << cluster_id << " {\n";
      file << "    label=\"Cluster " << cluster_id << "\";\n";
      for (int ii = 0; ii < n_points; ++ii) {
        if (cluster_index[ii] == cluster_id) {
          if (is_seed[ii] == 1) {
            file << "    " << ii << " [shape=doublecircle];\n";
          } else {
            file << "    " << ii << ";\n";
          }
        }
      }
      file << "  }\n";
    }

    // Directed edges following the nearest-higher-density relation
    for (int ii = 0; ii < n_points; ++ii) {
      const auto nearest_higher_idx = nearest_higher[ii];
      if (nearest_higher_idx >= 0 && nearest_higher_idx != ii) {
        file << "  " << ii << " -> " << nearest_higher_idx << ";\n";
      }
    }

    file << "}\n";
  }

}  // namespace clue
