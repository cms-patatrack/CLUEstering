/// @file SpatialGrid.hpp
/// @brief Read-only description of the spatial grid built by the Clusterer in its last run
/// @authors Simone Balducci

#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

CLUE_NAMESPACE_BEGIN

/// @brief Read-only description of the spatial grid built by the Clusterer in its last run
///
/// The grid is regular over the clustering space. The cells map each grid cell to the points it
/// contains, in CSR form: the points of cell `c` are the values in `[keys[c], keys[c + 1])`, and
/// the values are the positions of the points in the input given to `make_clusters`.
///
/// Geometry:
/// - along dimension `d` the grid covers `[min[d], max[d])` with `n_cells[d]` cells of size
///   `cell_size[d]`. A point goes in cell `(coord - min[d]) / cell_size[d]`, truncated and
///   clamped to `[0, n_cells[d] - 1]`, so points outside the range fall in the border cells.
/// - a periodic dimension covers one period, `[0, period)`, and its coordinates are wrapped into
///   that interval before binning.
/// - the global cell index is row-major: the last dimension varies fastest.
/// - in a batched run the grid is the same for all the batch items and the keys are concatenated,
///   so the cells of item `e` have global indexes from `e * total` to `(e + 1) * total - 1`,
///   where `total` is the product of `n_cells`.
///
/// @tparam Ndim The number of dimensions
/// @tparam TData The floating-point type of the coordinates
/// @tparam TDev The device holding the cell associations
///
/// @note The object refers to the buffers of the Clusterer, which are overwritten by the next
/// `make_clusters` call and freed when the Clusterer is destroyed.
/// @note The grid is filled asynchronously on the clustering queue. Copy it on that same queue,
/// or wait for it first.
/// @note Use `cells.extents()` for the number of keys and values: the underlying buffers can be
/// larger than the data they hold.
template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
struct SpatialGrid {
  /// @brief Map from each global cell index to the indexes of the points it contains
  const AssociationMap<TDev>& cells;
  /// @brief Lower edge of the grid along each dimension
  std::array<TData, Ndim> min;
  /// @brief Upper edge of the grid along each dimension
  std::array<TData, Ndim> max;
  /// @brief Size of the cells along each dimension
  std::array<TData, Ndim> cell_size;
  /// @brief Number of cells along each dimension
  std::array<std::int32_t, Ndim> n_cells;
  /// @brief Whether each dimension is periodic
  std::array<bool, Ndim> periodic;
  /// @brief Number of batch items in the last run, 1 for a run that is not batched
  std::size_t batch_size;
};

CLUE_NAMESPACE_END
