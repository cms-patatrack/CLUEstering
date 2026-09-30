
#pragma once

#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace clue::detail {

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>()))>
  void setup_tiles_from_extents(
      TQueue& queue,
      int32_t npoints,
      const clue::host_buffer<internal::CoordinateExtremes<Ndim, TData>>& min_max,
      std::optional<internal::Tiles<Ndim, TData, TDev>>& tiles,
      TData tile_edge,
      const std::array<uint8_t, Ndim>& wrapped_coordinates,
      std::size_t batch_size) {
    const auto grid = detail::compute_tile_grid(*min_max.data(), tile_edge, npoints, batch_size);

    if (!tiles.has_value()) {
      tiles = std::make_optional<internal::Tiles<Ndim, TData, TDev>>(queue, npoints, grid, batch_size);
    }
    // check if tiles are large enough for current data
    // before: compared keys to ntiles. I think extents is
    // ntiles * batch_size so we should compae to that
    // In CMSSW, we produce new tiles every event i think so 
    // it didnt matter, but now with extents it does i think
    // ALSO: this resets the extents so it is the 
    // last requested size, not the buffer capacity. Not a correctness
    // bug but will cause more initialises than needed
    if ((tiles->extents().values < static_cast<std::size_t>(npoints)) or
        (tiles->extents().keys < static_cast<std::size_t>(grid.ntiles) * batch_size)) {
      tiles->initialize(queue, npoints, grid, batch_size);
    } else {
      tiles->reset(npoints, grid, batch_size);
    }

    auto tile_sizes = clue::make_host_buffer<TData[Ndim]>(queue);
    std::copy(grid.tilesizes.begin(), grid.tilesizes.end(), tile_sizes.data());

    alpaka::memcpy(queue, tiles->minMax(), min_max);
    alpaka::memcpy(queue, tiles->tileSize(), tile_sizes);
    alpaka::memcpy(queue, tiles->wrapped(), clue::make_host_view(wrapped_coordinates.data(), Ndim));
    alpaka::wait(queue);
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TInput,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>()))>
  void setup_tiles(TQueue& queue,
                   const PointsHost<Ndim, TInput>& points,
                   std::optional<internal::Tiles<Ndim, std::remove_cv_t<TInput>, TDev>>& tiles,
                   std::remove_cv_t<TInput> tile_edge,
                   const std::array<uint8_t, Ndim>& wrapped_coordinates,
                   std::size_t batch_size = 1) {
    auto min_max =
        clue::make_host_buffer<internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>>(queue);
    detail::compute_extents(min_max.data(), points);
    setup_tiles_from_extents(
        queue, points.size(), min_max, tiles, tile_edge, wrapped_coordinates, batch_size);
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TInput,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>()))>
  void setup_tiles(TQueue& queue,
                   const PointsDevice<Ndim, TInput, TDev>& points,
                   std::optional<internal::Tiles<Ndim, std::remove_cv_t<TInput>, TDev>>& tiles,
                   std::remove_cv_t<TInput> tile_edge,
                   const std::array<uint8_t, Ndim>& wrapped_coordinates,
                   std::size_t batch_size = 1) {
    auto min_max =
        clue::make_host_buffer<internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>>(queue);
    detail::compute_extents(min_max.data(), points);
    setup_tiles_from_extents(
        queue, points.size(), min_max, tiles, tile_edge, wrapped_coordinates, batch_size);
  }

}  // namespace clue::detail
