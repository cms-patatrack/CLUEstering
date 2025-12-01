
#pragma once

#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace clue::detail {

  template <concepts::queue TQueue,
            std::size_t Ndim,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>()))>
  void setup_tiles(TQueue& queue,
                   std::optional<internal::Tiles<Ndim, TDev>>& tiles,
                   const PointsHost<Ndim>& points,
                   int points_per_tile,
                   const std::array<uint8_t, Ndim>& wrapped_coordinates) {
    // TODO: reconsider the way that we compute the number of tiles
    auto ntiles =
        static_cast<int32_t>(std::ceil(points.size() / static_cast<float>(points_per_tile)));
    const auto n_per_dim = static_cast<int32_t>(std::ceil(std::pow(ntiles, 1. / Ndim)));
    ntiles = static_cast<int32_t>(std::pow(n_per_dim, Ndim));

    if (!tiles.has_value()) {
      tiles = std::make_optional<internal::Tiles<Ndim, TDev>>(queue, points.size(), ntiles);
    }
    // check if tiles are large enough for current data
    if ((tiles->extents().values < static_cast<std::size_t>(points.size())) or
        (tiles->extents().keys < static_cast<std::size_t>(ntiles))) {
      tiles->initialize(queue, points.size(), ntiles, n_per_dim);
    } else {
      tiles->reset(points.size(), ntiles, n_per_dim);
    }

    auto min_max = clue::make_host_buffer<internal::CoordinateExtremes<Ndim>>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);
    detail::compute_tile_size(min_max.data(), tile_sizes.data(), points, n_per_dim);

    alpaka::memcpy(queue, tiles->minMax(), min_max);
    alpaka::memcpy(queue, tiles->tileSize(), tile_sizes);
    alpaka::memcpy(queue, tiles->wrapped(), clue::make_host_view(wrapped_coordinates.data(), Ndim));
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>()))>
  void setup_tiles(TQueue& queue,
                   std::optional<internal::Tiles<Ndim, TDev>>& tiles,
                   const PointsDevice<Ndim, TDev>& points,
                   int points_per_tile,
                   const std::array<uint8_t, Ndim>& wrapped_coordinates) {
    auto ntiles =
        static_cast<int32_t>(std::ceil(points.size() / static_cast<float>(points_per_tile)));
    const auto n_per_dim = static_cast<int32_t>(std::ceil(std::pow(ntiles, 1. / Ndim)));
    ntiles = static_cast<int32_t>(std::pow(n_per_dim, Ndim));

    if (!tiles.has_value()) {
      tiles = std::make_optional<internal::Tiles<Ndim, TDev>>(queue, points.size(), ntiles);
    }
    // check if tiles are large enough for current data
    if ((tiles->extents().values < static_cast<std::size_t>(points.size())) or
        (tiles->extents().keys < static_cast<std::size_t>(ntiles))) {
      tiles->initialize(queue, points.size(), ntiles, n_per_dim);
    } else {
      tiles->reset(points.size(), ntiles, n_per_dim);
    }

    auto min_max = clue::make_host_buffer<internal::CoordinateExtremes<Ndim>>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);
    detail::compute_tile_size(min_max.data(), tile_sizes.data(), points, n_per_dim);

    alpaka::memcpy(queue, tiles->minMax(), min_max);
    alpaka::memcpy(queue, tiles->tileSize(), tile_sizes);
    alpaka::memcpy(queue, tiles->wrapped(), clue::make_host_view(wrapped_coordinates.data(), Ndim));
  }

}  // namespace clue::detail
