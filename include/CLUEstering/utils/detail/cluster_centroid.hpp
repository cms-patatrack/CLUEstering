
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/internal/nostd/zip_iterator.hpp"
#include "CLUEstering/utils/cluster_centroid.hpp"
#include "CLUEstering/utils/get_clusters.hpp"
#include <algorithm>
#include <array>
#include <vector>

namespace clue {

  template <std::size_t Ndim>
  inline Centroid<Ndim> cluster_centroid(const clue::PointsHost<Ndim>& points,
                                         std::size_t cluster_id) {
    assert(points.clustered());
    auto cluster_ids = points.clusterIndexes();
    auto clusters = get_clusters(points);
    // TODO: add error handling
    Centroid<Ndim> centroid;
    auto size = clusters.count(cluster_id);
    for (auto dim = 0u; dim < Ndim; ++dim) {
      auto coords = points.coords(dim);
      std::for_each_n(nostd::zip(coords.begin(), cluster_ids.begin()),
                      points.size(),
                      [=, &centroid](auto&& tuple) -> void {
                        const auto coord = std::get<0>(tuple);
                        const auto point_cluster = std::get<1>(tuple);
                        if (static_cast<std::size_t>(point_cluster) == cluster_id) {
                          centroid[dim] += coord;
                        }
                      });
      centroid[dim] /= static_cast<float>(size);
    }

    return centroid;
  }

  template <std::size_t Ndim>
  inline Centroids<Ndim> cluster_centroids(const clue::PointsHost<Ndim>& points) {
    assert(points.clustered());
    auto cluster_ids = points.clusterIndexes();
    auto clusters = get_clusters(points);
    const auto n_clusters = clusters.size();

    Centroids<Ndim> centroids(n_clusters);
    for (auto dim = 0u; dim < Ndim; ++dim) {
      auto coords = points.coords(dim);
      std::for_each_n(nostd::zip(coords.begin(), cluster_ids.begin()),
                      points.size(),
                      [=, &centroids](auto&& tuple) -> void {
                        const auto coord = std::get<0>(tuple);
                        const auto point_cluster = std::get<1>(tuple);
                        if (point_cluster >= 0)
                          centroids[point_cluster][dim] += coord;
                      });
      std::ranges::for_each(centroids, [&, dim, cl = 0](auto& centroid) mutable {
        const auto size = clusters.count(cl);
        centroid[dim] /= static_cast<float>(size);
        ++cl;
      });
    }

    return centroids;
  }

}  // namespace clue
