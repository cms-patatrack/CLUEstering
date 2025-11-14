
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include "CLUEstering/internal/algorithm/default_policy.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include "CLUEstering/internal/nostd/minimum.hpp"
#include <algorithm>
#include <execution>

namespace clue {
  namespace detail {

    template <std::size_t Ndim>
    void compute_tile_size(const clue::PointsHost<Ndim>& h_points,
                           internal::CoordinateExtremes<Ndim>& min_max,
                           std::array<float, Ndim>& tile_sizes,
                           int32_t nPerDim) {
      for (auto dim = 0u; dim < Ndim; ++dim) {
        auto coords = h_points.coords(dim);
        const float dim_max = std::reduce(coords.begin(),
                                          coords.end(),
                                          std::numeric_limits<float>::lowest(),
                                          clue::nostd::maximum<float>{});
        const float dim_min = std::reduce(coords.begin(),
                                          coords.end(),
                                          std::numeric_limits<float>::max(),
                                          clue::nostd::minimum<float>{});
        min_max = {dim_min, dim_max};

        const auto tile_size = (dim_max - dim_min) / nPerDim;
        tile_sizes[dim] = tile_size;
      }
    }

    template <std::size_t Ndim>
    void compute_tile_size(Queue& queue,
                           const clue::PointsDevice<Ndim>& dev_points,
                           internal::CoordinateExtremes<Ndim>& min_max,
                           std::array<float, Ndim>& tile_sizes,
                           uint32_t nPerDim) {
      for (auto dim = 0u; dim < Ndim; ++dim) {
        auto coords = dev_points.coords(dim);
        const auto dim_max = clue::internal::algorithm::reduce(coords.begin(),
                                                               coords.end(),
                                                               std::numeric_limits<float>::lowest(),
                                                               clue::nostd::maximum<float>{});
        const auto dim_min = clue::internal::algorithm::reduce(coords.begin(),
                                                               coords.end(),
                                                               std::numeric_limits<float>::max(),
                                                               clue::nostd::minimum<float>{});
        min_max = {dim_min, dim_max};

        const auto tile_size = (dim_max - dim_min) / nPerDim;
        tile_sizes[dim] = tile_size;
      }
    }

  }  // namespace detail
}  // namespace clue
