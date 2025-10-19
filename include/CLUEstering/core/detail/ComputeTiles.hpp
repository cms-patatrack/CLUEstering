
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/Tiles.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include "CLUEstering/internal/algorithm/default_policy.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include "CLUEstering/internal/nostd/minimum.hpp"
#include <algorithm>
#include <execution>

namespace clue {
  namespace detail {

    template <std::size_t Ndim>
    void compute_tile_size(clue::CoordinateExtremes<Ndim>* min_max,
                           float* tile_sizes,
                           const clue::PointsHost<Ndim>& h_points,
                           int32_t nPerDim) {
      for (size_t dim{}; dim != Ndim; ++dim) {
        auto coords = h_points.coords(dim);
        const float dimMax = std::reduce(clue::internal::default_policy,
                                         coords.begin(),
                                         coords.end(),
                                         std::numeric_limits<float>::lowest(),
                                         clue::nostd::maximum<float>{});
        const float dimMin = std::reduce(clue::internal::default_policy,
                                         coords.begin(),
                                         coords.end(),
                                         std::numeric_limits<float>::max(),
                                         clue::nostd::minimum<float>{});

        min_max->min(dim) = dimMin;
        min_max->max(dim) = dimMax;

        const float tileSize = (dimMax - dimMin) / nPerDim;
        tile_sizes[dim] = tileSize;
      }
    }

    template <std::size_t Ndim>
    void compute_tile_size(Queue& queue,
                           clue::CoordinateExtremes<Ndim>* min_max,
                           float* tile_sizes,
                           const clue::PointsDevice<Ndim>& dev_points,
                           uint32_t nPerDim) {
      for (size_t dim{}; dim != Ndim; ++dim) {
        auto coords = dev_points.coords(dim);
        const auto dimMax = clue::internal::algorithm::reduce(coords.begin(),
                                                              coords.end(),
                                                              std::numeric_limits<float>::lowest(),
                                                              clue::nostd::maximum<float>{});
        const auto dimMin = clue::internal::algorithm::reduce(coords.begin(),
                                                              coords.end(),
                                                              std::numeric_limits<float>::max(),
                                                              clue::nostd::minimum<float>{});

        min_max->min(dim) = dimMin;
        min_max->max(dim) = dimMax;

        const float tileSize = (dimMax - dimMin) / nPerDim;
        tile_sizes[dim] = tileSize;
      }
    }

  }  // namespace detail
}  // namespace clue
