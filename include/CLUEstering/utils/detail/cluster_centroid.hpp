
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/internal/nostd/zip_iterator.hpp"
#include "CLUEstering/utils/cluster_centroid.hpp"
#include <algorithm>
#include <array>
#include <vector>

namespace clue {

  template <uint8_t Ndim>
  using Centroid = std::array<float, Ndim>;

  template <uint8_t Ndim>
  using Centroids = std::vector<Centroid>;

  template <uint8_t Ndim>
  inline Centroid<Ndim> cluster_centroid(const clue::PointsHost<Ndim>& points,
                                         std::size_t cluster_id) {
    // TODO: add error handling
    // TODO: possibly use getClusters inside here
    Centroid centroid;
    int size = 0;
    for (auto dim = 0; dim < Ndim; ++dim) {
      auto coords = points.coords(dim);
      auto cluster_ids = points.clusterIndexes();
      std::for_each_n(nostd::zip(coords.begin(), cluster_ids.begin()),
                      [&centroid, dim, &size](auto&& tuple) -> void {
                        const auto coord = std::get<0>(tuple);
                        const auto point_cluster = std::get<1>(tuple);
                        if (point_cluster == cluster_id) {
                          centroid[dim] += coord;
                          if (dim == 0)
                            ++size;
                        }
                      });
    }
  }

}  // namespace clue
