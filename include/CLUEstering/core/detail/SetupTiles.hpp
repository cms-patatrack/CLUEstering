
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/default_allocator.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>

CLUE_NAMESPACE_BEGIN
namespace detail {

  /// @brief Period of each wrapped coordinate for the tiles, 0 for a coordinate that does not wrap
  ///
  /// The period comes from the metric, which must have periods(), as PeriodicEuclideanMetric.
  /// A wrapped coordinate without a positive period throws std::invalid_argument.
  template <std::size_t Ndim, std::floating_point TData, typename TMetric>
  std::array<TData, Ndim> tile_periods(const std::array<uint8_t, Ndim>& wrapped_coordinates,
                                       const TMetric& metric) {
    std::array<TData, Ndim> periods{};
    for (auto dim = 0u; dim != Ndim; ++dim) {
      if (not wrapped_coordinates[dim])
        continue;
      if constexpr (requires { metric.periods(); })
        periods[dim] = metric.periods()[dim] > TData{0} ? metric.periods()[dim] : TData{0};
      if (periods[dim] == TData{0})
        throw std::invalid_argument(
            "A wrapped coordinate needs a metric with a positive period for it, such as "
            "PeriodicEuclideanMetric.");
    }
    return periods;
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>())),
            concepts::allocator TAllocator = DefaultAllocator>
  void setup_tiles_from_extents(TQueue& queue,
                                int32_t npoints,
                                internal::CoordinateExtremes<Ndim, TData>& extremes,
                                host_buffer<TData[Ndim]>& tile_sizes,
                                std::optional<internal::Tiles<Ndim, TData, TDev>>& tiles,
                                TData tile_edge,
                                const std::array<uint8_t, Ndim>& wrapped_coordinates,
                                const std::array<TData, Ndim>& periods,
                                std::size_t batch_size,
                                const TAllocator& allocator = TAllocator{}) {
    // The tiles of a wrapped coordinate cover one period, [0, period)
    for (auto dim = 0u; dim != Ndim; ++dim) {
      if (wrapped_coordinates[dim]) {
        extremes.min(dim) = TData{0};
        extremes.max(dim) = periods[dim];
      }
    }
    const auto grid = detail::compute_tile_grid(extremes, tile_edge, npoints, batch_size);

    if (!tiles.has_value()) {
      tiles = std::make_optional<internal::Tiles<Ndim, TData, TDev>>(
          queue, npoints, grid, batch_size, allocator);
    }
    // check if tiles are large enough for current data. The keys hold ntiles for every event in
    // the batch, so compare against ntiles * batch_size.
    // Note: reset() sets extents() to the requested size rather than the allocated capacity, so a
    // later larger input re-initialises even when the buffers would fit.
    if ((tiles->extents().values < static_cast<std::size_t>(npoints)) or
        (tiles->extents().keys < static_cast<std::size_t>(grid.ntiles) * batch_size)) {
      tiles->initialize(queue, npoints, grid, batch_size, allocator);
    } else {
      tiles->reset(npoints, grid, batch_size);
    }

    std::copy(grid.tilesizes.begin(), grid.tilesizes.end(), tile_sizes.data());

    alpaka::memcpy(queue, tiles->minMax(), alpaka::createView(host, &extremes, Scalar{}));
    alpaka::memcpy(queue, tiles->tileSize(), tile_sizes);
    alpaka::memcpy(
        queue, tiles->wrapped(), alpaka::createView(host, wrapped_coordinates.data(), Ndim));
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TInput,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>())),
            concepts::allocator TAllocator = DefaultAllocator>
  void setup_tiles(
      TQueue& queue,
      const PointsHost<Ndim, TInput>& points,
      std::optional<host_buffer<internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>>>&
          extremes,
      bool compute_extremes,
      std::optional<host_buffer<std::remove_cv_t<TInput>[Ndim]>>& tile_sizes,
      std::optional<internal::Tiles<Ndim, std::remove_cv_t<TInput>, TDev>>& tiles,
      std::remove_cv_t<TInput> tile_edge,
      const std::array<uint8_t, Ndim>& wrapped_coordinates,
      const std::array<std::remove_cv_t<TInput>, Ndim>& periods,
      std::size_t batch_size = 1,
      const TAllocator& allocator = TAllocator{}) {
    if (!extremes.has_value()) {
      extremes = std::make_optional(
          alpaka::allocMappedBuf<internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>>(
              host, platform<TQueue>(), Scalar{}));
    }
    if (!tile_sizes.has_value()) {
      tile_sizes = std::make_optional(
          alpaka::allocMappedBuf<std::remove_cv_t<TInput>>(host, platform<TQueue>(), Ndim));
    }
    if (compute_extremes)
      detail::compute_extents(extremes->data(), points);

    setup_tiles_from_extents(queue,
                             points.size(),
                             *extremes->data(),
                             *tile_sizes,
                             tiles,
                             tile_edge,
                             wrapped_coordinates,
                             periods,
                             batch_size,
                             allocator);
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TInput,
            concepts::device TDev = decltype(alpaka::getDev(std::declval<TQueue>())),
            concepts::allocator TAllocator = DefaultAllocator>
  void setup_tiles(
      TQueue& queue,
      const PointsDevice<Ndim, TInput, TDev>& points,
      std::optional<host_buffer<internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>>>&
          extremes,
      bool compute_extremes,
      std::optional<host_buffer<std::remove_cv_t<TInput>[Ndim]>>& tile_sizes,
      std::optional<internal::Tiles<Ndim, std::remove_cv_t<TInput>, TDev>>& tiles,
      std::remove_cv_t<TInput> tile_edge,
      const std::array<uint8_t, Ndim>& wrapped_coordinates,
      const std::array<std::remove_cv_t<TInput>, Ndim>& periods,
      std::size_t batch_size = 1,
      const TAllocator& allocator = TAllocator{}) {
    if (!extremes.has_value()) {
      extremes = std::make_optional(
          alpaka::allocMappedBuf<internal::CoordinateExtremes<Ndim, std::remove_cv_t<TInput>>>(
              host, platform<TQueue>(), Scalar{}));
    }
    if (!tile_sizes.has_value()) {
      tile_sizes = std::make_optional(
          alpaka::allocMappedBuf<std::remove_cv_t<TInput>>(host, platform<TQueue>(), Ndim));
    }

    if (compute_extremes)
      detail::compute_extents(queue, extremes->data(), points);

    setup_tiles_from_extents(queue,
                             points.size(),
                             *extremes->data(),
                             *tile_sizes,
                             tiles,
                             tile_edge,
                             wrapped_coordinates,
                             periods,
                             batch_size,
                             allocator);
  }

}  // namespace detail
CLUE_NAMESPACE_END
