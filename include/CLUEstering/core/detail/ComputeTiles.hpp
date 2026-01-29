
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include "CLUEstering/internal/nostd/minimum.hpp"
#include <algorithm>
#include <concepts>

namespace clue::detail {

  template <std::size_t Ndim, std::floating_point TData>
  void compute_tile_size(internal::CoordinateExtremes<Ndim, TData>* min_max,
                         TData* tile_sizes,
                         const clue::PointsHost<Ndim, TData>& h_points,
                         std::int32_t nPerDim) {
    for (auto dim = 0u; dim != Ndim; ++dim) {
      auto coords = h_points.coords(dim);
      const auto dimMax = std::reduce(coords.begin(),
                                      coords.end(),
                                      std::numeric_limits<TData>::lowest(),
                                      clue::nostd::maximum<TData>{});
      const auto dimMin = std::reduce(coords.begin(),
                                      coords.end(),
                                      std::numeric_limits<TData>::max(),
                                      clue::nostd::minimum<TData>{});

      min_max->min(dim) = dimMin;
      min_max->max(dim) = dimMax;

      const auto tileSize = (dimMax - dimMin) / nPerDim;
      tile_sizes[dim] = tileSize;
    }
  }

  template <std::size_t Ndim, std::floating_point TData>
  void compute_tile_size(internal::CoordinateExtremes<Ndim, TData>* min_max,
                         TData* tile_sizes,
                         const clue::PointsDevice<Ndim, TData>& dev_points,
                         std::uint32_t nPerDim) {
    for (auto dim = 0u; dim != Ndim; ++dim) {
      auto coords = dev_points.coords(dim);
      const auto dimMax = clue::internal::algorithm::reduce(coords.begin(),
                                                            coords.end(),
                                                            std::numeric_limits<TData>::lowest(),
                                                            clue::nostd::maximum<TData>{});
      const auto dimMin = clue::internal::algorithm::reduce(coords.begin(),
                                                            coords.end(),
                                                            std::numeric_limits<TData>::max(),
                                                            clue::nostd::minimum<TData>{});

      min_max->min(dim) = dimMin;
      min_max->max(dim) = dimMax;

      const auto tileSize = (dimMax - dimMin) / nPerDim;
      tile_sizes[dim] = tileSize;
    }
  }

}  // namespace clue::detail
