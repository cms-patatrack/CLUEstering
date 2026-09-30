
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include "CLUEstering/internal/nostd/minimum.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

namespace clue::detail {

  /// @brief An upper bound on the number of tiles per point
  ///
  /// At most set 4 * number of points tiles. Avoids creating lots of empty tiles if points are sparse
  /// in comparison to the tile edge computed from search radii. 
  /// TODO: tune to see optimal parameter for this
  inline constexpr std::size_t max_tiles_per_point = 4;

  /// @brief Tile edge derived from the clustering radii
  ///
  /// The search box is +- radius along every coordinate, so the same edge is used along every
  /// dimension. 
  /// If dr <= od <= 4*dr, then edge is dr and the density searc spans 3 tiles per dimension,
  /// nearest-higher searches 9 tiles, at most
  /// If dr < 4*dr, then nearest-higher spans 9 tiles per dimension, and density at most 3
  /// else: both span 3
  /// This assume tiles of exactly this edge. compute_tile_grid rounds the number of tiles up, 
  /// which makes the tiles slightly shorter and can add one tile per dimension
  template <std::floating_point TData>
  constexpr TData tile_edge(TData density_radius, TData outlier_distance) {
    return std::min(std::max(density_radius, outlier_distance / TData{4}), outlier_distance);
  }

  template <std::size_t Ndim, std::floating_point TInput>
  void compute_extents(internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>* min_max,
                       const clue::PointsHost<Ndim, TInput>& h_points) {
    for (auto dim = 0u; dim != Ndim; ++dim) {
      auto coords = h_points.coords(dim);
      min_max->max(dim) = std::reduce(coords.begin(),
                                      coords.end(),
                                      std::numeric_limits<TInput>::lowest(),
                                      clue::nostd::maximum<TInput>{});
      min_max->min(dim) = std::reduce(coords.begin(),
                                      coords.end(),
                                      std::numeric_limits<TInput>::max(),
                                      clue::nostd::minimum<TInput>{});
    }
  }

  template <std::size_t Ndim, std::floating_point TInput>
  void compute_extents(internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>* min_max,
                       const clue::PointsDevice<Ndim, TInput>& dev_points) {
    for (auto dim = 0u; dim != Ndim; ++dim) {
      auto coords = dev_points.coords(dim);
      min_max->max(dim) = clue::internal::algorithm::reduce(coords.begin(),
                                                            coords.end(),
                                                            std::numeric_limits<TInput>::lowest(),
                                                            clue::nostd::maximum<TInput>{});
      min_max->min(dim) = clue::internal::algorithm::reduce(coords.begin(),
                                                            coords.end(),
                                                            std::numeric_limits<TInput>::max(),
                                                            clue::nostd::minimum<TInput>{});
    }
  }

  /// @brief Compute the tile grid for the given extents
  ///
  /// Along each dimension the number of tiles is the extent divided by the tile edge, rounded up.
  /// (rounded up means the above 3 and 9 tiles per dim are approximates, can be 4 or 10)
  /// If the total number of tiles over the batch exceeds `max_tiles_per_point` per point, the
  /// edge is enlarged uniformly until it does not.
  template <std::size_t Ndim, std::floating_point TData>
  internal::TileGrid<Ndim, TData> compute_tile_grid(
      const internal::CoordinateExtremes<Ndim, TData>& min_max,
      TData tile_edge,
      std::size_t npoints,
      std::size_t batch_size) {

    const auto max_tiles = 
      static_cast<double>(std::max(max_tiles_per_point * npoints, std::size_t{1})) / batch_size;

    auto edge = static_cast<double>(tile_edge);
    std::array<double, Ndim> nperdim{};
    double ntiles;
    while (true) {
      ntiles = 1.;
      for (auto dim = 0u; dim != Ndim; ++dim) {
        nperdim[dim] = std::max(1., std::ceil(static_cast<double>(min_max.range(dim)) / edge));
        ntiles *= nperdim[dim];
      }
      if (ntiles <= max_tiles or ntiles == 1.)
        break;
      edge *= std::pow(ntiles / max_tiles, 1. / Ndim);
    }

    internal::TileGrid<Ndim, TData> grid;
    grid.ntiles = static_cast<int32_t>(ntiles);
    for (auto dim = 0u; dim != Ndim; ++dim) {
      grid.nperdim[dim] = static_cast<int32_t>(nperdim[dim]);
      const auto range = min_max.range(dim);
      // A dimension with 0 variance gets a single tile of any positive size, so that every coordinate
      // falls in bin 0.
      grid.tilesizes[dim] =
          range > TData{0} ? range / static_cast<TData>(nperdim[dim]) : TData{1};
    }
    return grid;
  }

}  // namespace clue::detail
