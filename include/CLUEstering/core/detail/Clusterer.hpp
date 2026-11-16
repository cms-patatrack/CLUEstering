
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/BatchedClusteringKernels.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/core/detail/MergeClusters.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/detail/SetupPoints.hpp"
#include "CLUEstering/core/detail/SetupTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/internal/nostd/ceil_div.hpp"
#include "CLUEstering/utils/get_clusters.hpp"

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>

CLUE_NAMESPACE_BEGIN

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
Clusterer<Ndim, DataType, AllocatorType>::Clusterer(AllocatorType allocator,
                                                    value_type density_radius,
                                                    value_type min_density,
                                                    std::optional<value_type> outlier_distance,
                                                    std::optional<value_type> seeding_distance)
    : m_density_radius{density_radius},
      m_seeding_distance{seeding_distance.value_or(density_radius)},
      m_min_density{min_density},
      m_outlier_distance{outlier_distance.value_or(density_radius)},
      m_wrappedCoordinates{},
      m_allocator{std::move(allocator)} {
  if (m_density_radius <= static_cast<value_type>(0.) ||
      m_min_density < static_cast<value_type>(0.) ||
      m_outlier_distance <= static_cast<value_type>(0.) ||
      m_seeding_distance <= static_cast<value_type>(0.)) {
    throw std::invalid_argument("Invalid clustering parameters. The parameters must be positive.");
  }
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
Clusterer<Ndim, DataType, AllocatorType>::Clusterer(value_type density_radius,
                                                    value_type min_density,
                                                    std::optional<value_type> outlier_distance,
                                                    std::optional<value_type> seeding_distance)
  requires std::default_initializable<AllocatorType>
    : Clusterer(AllocatorType{}, density_radius, min_density, outlier_distance, seeding_distance) {}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
void Clusterer<Ndim, DataType, AllocatorType>::setParameters(
    value_type density_radius,
    value_type min_density,
    std::optional<value_type> outlier_distance,
    std::optional<value_type> seeding_distance) {
  m_density_radius = density_radius;
  m_outlier_distance = outlier_distance.value_or(density_radius);
  m_seeding_distance = seeding_distance.value_or(density_radius);
  m_min_density = min_density;

  if (m_density_radius <= static_cast<value_type>(0.) ||
      m_min_density < static_cast<value_type>(0.) ||
      m_outlier_distance <= static_cast<value_type>(0.) ||
      m_seeding_distance <= static_cast<value_type>(0.)) {
    throw std::invalid_argument("Invalid clustering parameters. The parameters must be positive.");
  }
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
void Clusterer<Ndim, DataType, AllocatorType>::setMergeThreshold(
    std::optional<value_type> merge_threshold) {
  if (merge_threshold.has_value() && !(*merge_threshold > static_cast<value_type>(0.) &&
                                       *merge_threshold <= static_cast<value_type>(1.))) {
    throw std::invalid_argument("Invalid merge threshold. It must be in the interval (0, 1].");
  }
  m_merge_threshold = merge_threshold;
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::ranges::contiguous_range TRange>
void Clusterer<Ndim, DataType, AllocatorType>::setExtremes(const TRange& extremes)
  requires std::convertible_to<std::ranges::range_value_t<TRange>, std::pair<value_type, value_type>>
{
  if (std::ranges::size(extremes) != Ndim) {
    throw std::invalid_argument("The number of extremes must match the number of dimensions");
  }
  const auto* ranges = std::ranges::data(extremes);
  for (auto dim = 0u; dim != Ndim; ++dim) {
    const std::pair<value_type, value_type> range = ranges[dim];
    if (!(range.first < range.second)) {
      throw std::invalid_argument(
          "The minimum of each coordinate must be smaller than its maximum");
    }
  }

  if (!m_extremes.has_value()) {
    m_extremes =
        std::make_optional(alpaka::allocMappedBuf<internal::CoordinateExtremes<Ndim, value_type>>(
            host, platform<Platform>(), Scalar{}));
  }
  for (auto dim = 0u; dim != Ndim; ++dim) {
    const std::pair<value_type, value_type> range = ranges[dim];
    m_extremes->data()->min(dim) = range.first;
    m_extremes->data()->max(dim) = range.second;
  }
  m_extremes_set = true;
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
void Clusterer<Ndim, DataType, AllocatorType>::setExtremes(const extremes_type& extremes) {
  setExtremes(std::span{extremes});
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
inline void Clusterer<Ndim, DataType, AllocatorType>::make_clusters(
    Queue& queue,
    clue::PointsHost<Ndim, InputType>& h_points,
    const DistanceMetric& metric,
    const Kernel& kernel) {
  auto d_points = detail::make_device_points(queue, h_points, m_allocator);

  setup(queue, h_points, detail::tile_periods<Ndim, value_type>(m_wrappedCoordinates, metric));
  make_clusters_impl(d_points, metric, kernel, queue);
  detail::sync_results(queue, h_points, d_points);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
inline void Clusterer<Ndim, DataType, AllocatorType>::make_clusters(
    Queue& queue,
    clue::PointsHost<Ndim, InputType>& h_points,
    clue::PointsDevice<Ndim, value_type>& dev_points,
    const DistanceMetric& metric,
    const Kernel& kernel) {
  setup(queue, h_points, detail::tile_periods<Ndim, value_type>(m_wrappedCoordinates, metric));
  clue::copyToDevice(queue, dev_points, h_points, m_allocator);
  make_clusters_impl(dev_points, metric, kernel, queue);
  clue::copyToHost(queue, h_points, dev_points);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
inline void Clusterer<Ndim, DataType, AllocatorType>::make_clusters(
    Queue& queue,
    clue::PointsDevice<Ndim, InputType>& dev_points,
    const DistanceMetric& metric,
    const Kernel& kernel) {
  detail::setup_tiles(queue,
                      dev_points,
                      m_extremes,
                      !m_extremes_set,
                      m_tile_size,
                      m_tiles,
                      detail::tile_edge(m_density_radius, m_outlier_distance),
                      m_wrappedCoordinates,
                      detail::tile_periods<Ndim, value_type>(m_wrappedCoordinates, metric),
                      1,
                      m_allocator);
  make_clusters_impl(dev_points, metric, kernel, queue);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
inline void Clusterer<Ndim, DataType, AllocatorType>::make_clusters(
    Queue& queue,
    clue::PointsHost<Ndim, InputType>& h_points,
    clue::PointsDevice<Ndim, value_type>& dev_points,
    std::span<const uint32_t> batch_item_sizes,
    const DistanceMetric& metric,
    const Kernel& kernel) {
  const auto batch_size = batch_item_sizes.size();
  setup_batch(queue,
              h_points,
              batch_size,
              detail::tile_periods<Ndim, value_type>(m_wrappedCoordinates, metric));
  clue::copyToDevice(queue, dev_points, h_points, m_allocator);
  make_clusters_batched(dev_points, batch_item_sizes, metric, kernel, queue);
  clue::copyToHost(queue, h_points, dev_points);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
inline void Clusterer<Ndim, DataType, AllocatorType>::make_clusters(
    Queue& queue,
    clue::PointsHost<Ndim, InputType>& h_points,
    std::span<const uint32_t> batch_item_sizes,
    const DistanceMetric& metric,
    const Kernel& kernel) {
  auto d_points = detail::make_device_points(queue, h_points, m_allocator);

  const auto batch_size = batch_item_sizes.size();
  setup_batch(queue,
              h_points,
              batch_size,
              detail::tile_periods<Ndim, value_type>(m_wrappedCoordinates, metric));
  make_clusters_batched(d_points, batch_item_sizes, metric, kernel, queue);
  detail::sync_results(queue, h_points, d_points);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
inline void Clusterer<Ndim, DataType, AllocatorType>::make_clusters(
    Queue& queue,
    clue::PointsDevice<Ndim, InputType>& dev_points,
    std::span<const uint32_t> batch_item_sizes,
    const DistanceMetric& metric,
    const Kernel& kernel) {
  const auto batch_size = batch_item_sizes.size();
  setup_batch(queue,
              dev_points,
              batch_size,
              detail::tile_periods<Ndim, value_type>(m_wrappedCoordinates, metric));
  make_clusters_batched(dev_points, batch_item_sizes, metric, kernel, queue);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::ranges::contiguous_range TRange>
  requires std::integral<std::ranges::range_value_t<TRange>>
inline void Clusterer<Ndim, DataType, AllocatorType>::setWrappedCoordinates(
    const TRange& wrapped_coordinates) {
  std::ranges::copy(wrapped_coordinates | std::views::take(Ndim), m_wrappedCoordinates.begin());
}
template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::integral... TArgs>
inline void Clusterer<Ndim, DataType, AllocatorType>::setWrappedCoordinates(
    TArgs... wrappedCoordinates) {
  m_wrappedCoordinates = {static_cast<uint8_t>(wrappedCoordinates)...};
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
inline std::span<const int32_t> Clusterer<Ndim, DataType, AllocatorType>::getSeeds() const {
  throw std::runtime_error(
      "getSeeds() is deprecated and no longer provides seeds: use getSeeds(queue, d_points) or "
      "getSeeds(h_points).");
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
inline SpatialGrid<Ndim, typename Clusterer<Ndim, DataType, AllocatorType>::value_type, Device>
Clusterer<Ndim, DataType, AllocatorType>::getSpatialGrid() const {
  if (!m_tiles.has_value()) {
    throw std::runtime_error("The spatial grid is not available. Please run make_clusters first.");
  }
  return m_tiles->spatialGrid();
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType>
inline auto Clusterer<Ndim, DataType, AllocatorType>::getSeeds(
    Queue& queue, clue::PointsDevice<Ndim, InputType>& d_points) {
  if (!d_points.clustered()) {
    throw std::runtime_error("Seeds are not available. Please run make_clusters first.");
  }
  constexpr std::size_t block_size = 256;

  const auto nclusters = static_cast<std::size_t>(d_points.n_clusters());
  auto seeds = internal::alloc_device_buffer<int32_t>(queue, nclusters, m_allocator);
  if (nclusters > 0) {
    detail::collectSeeds<internal::Acc>(queue, block_size, d_points.view(), seeds.data());
  }
  return seeds;
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType>
inline std::vector<int32_t> Clusterer<Ndim, DataType, AllocatorType>::getSeeds(
    const clue::PointsHost<Ndim, InputType>& h_points) {
  if (!h_points.clustered()) {
    throw std::runtime_error("Seeds are not available. Please run make_clusters first.");
  }
  if (!h_points.has_seeds()) {
    throw std::runtime_error(
        "Seeds are not available on the host. Call store_seeds on the host points before "
        "clustering.");
  }
  const auto cluster_index = h_points.clusterIndexes();
  const auto seed_flags = h_points.seed_flags();
  const auto nclusters = detail::compute_nclusters(cluster_index);
  std::vector<int32_t> seeds(static_cast<std::size_t>(nclusters > 0 ? nclusters : 0), -1);
  for (std::size_t i = 0; i < cluster_index.size(); ++i) {
    if (seed_flags[i] != 0 && cluster_index[i] >= 0) {
      seeds[cluster_index[i]] = static_cast<int32_t>(i);
    }
  }
  return seeds;
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType>
inline AssociationMapHost Clusterer<Ndim, DataType, AllocatorType>::getClusters(
    const clue::PointsHost<Ndim, InputType>& h_points) {
  return clue::get_clusters(h_points);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType>
inline AssociationMap<Device> Clusterer<Ndim, DataType, AllocatorType>::getClusters(
    Queue& queue, const clue::PointsDevice<Ndim, InputType>& d_points) {
  return clue::get_clusters(queue, d_points, m_allocator);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType>
inline AssociationMapHost Clusterer<Ndim, DataType, AllocatorType>::getSampleAssociations(
    Queue& queue, clue::PointsHost<Ndim, InputType>& h_points) {
  if (!h_points.clustered()) {
    throw std::runtime_error(
        "Sample associations are not available. Please run make_clusters first.");
  }
  if (!m_event_offsets.has_value()) {
    throw std::runtime_error(
        "Sample associations are not available: the last clustering run was not batched.");
  }

  std::vector<std::size_t> event_offsets(m_batch_size + 1);
  alpaka::memcpy(
      queue, alpaka::createView(host, event_offsets.data(), m_batch_size + 1), *m_event_offsets);
  alpaka::wait(queue);

  const auto cluster_index = h_points.clusterIndexes();
  const auto n_clusters = detail::compute_nclusters(cluster_index);
  std::vector<std::int32_t> cluster_to_event(
      static_cast<std::size_t>(n_clusters > 0 ? n_clusters : 0), 0);
  for (std::size_t i = 0; i < cluster_index.size(); ++i) {
    // all the points of a cluster belong to the same batch item
    if (cluster_index[i] >= 0) {
      // binary search of the event e such that event_offsets[e] <= i < event_offsets[e+1]
      std::size_t lo = 0;
      std::size_t hi = m_batch_size;
      while (hi - lo > 1) {
        const auto mid = lo + (hi - lo) / 2;
        if (event_offsets[mid] <= i) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      cluster_to_event[cluster_index[i]] = static_cast<std::int32_t>(lo);
    }
  }
  return internal::make_associator(
      std::span<const std::int32_t>{cluster_to_event.data(), cluster_to_event.size()});
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType>
inline AssociationMap<Device> Clusterer<Ndim, DataType, AllocatorType>::getSampleAssociations(
    Queue& queue, clue::PointsDevice<Ndim, InputType>& d_points) {
  if (!d_points.clustered()) {
    throw std::runtime_error(
        "Sample associations are not available. Please run make_clusters first.");
  }
  if (!m_event_offsets.has_value()) {
    throw std::runtime_error(
        "Sample associations are not available: the last clustering run was not batched.");
  }
  constexpr std::size_t block_size = 256;

  const auto n_clusters = static_cast<std::size_t>(d_points.n_clusters());
  auto cluster_to_event =
      internal::alloc_device_buffer<std::int32_t>(queue, n_clusters, m_allocator);
  if (n_clusters > 0) {
    detail::eventOfSeeds<internal::Acc>(queue,
                                        block_size,
                                        d_points.view(),
                                        m_event_offsets->data(),
                                        m_batch_size,
                                        cluster_to_event.data());
  }
  auto associations = internal::make_associator(
      queue, std::span<const std::int32_t>{cluster_to_event.data(), n_clusters}, m_allocator);
  alpaka::wait(queue);
  return associations;
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
void Clusterer<Ndim, DataType, AllocatorType>::make_clusters_impl(
    clue::PointsDevice<Ndim, InputType>& dev_points,
    const DistanceMetric& metric,
    const Kernel& kernel,
    Queue& queue) {
  // reset in case the previous run was batched
  m_event_offsets.reset();
  m_batch_size = 0;

  constexpr std::size_t block_size = 256;
  m_tiles->template fill<internal::Acc>(queue, dev_points, m_allocator);

  const Idx grid_size = nostd::ceil_div(dev_points.size(), block_size);
  auto work_division = clue::make_workdiv<internal::Acc>(grid_size, block_size);

  detail::computeLocalDensity<internal::Acc>(
      queue, work_division, m_tiles->view(), dev_points.view(), kernel, m_density_radius, metric);

  if (!m_nclusters.has_value()) {
	m_nclusters.emplace(internal::alloc_device_buffer<std::size_t>(queue, Scalar{}, m_allocator));
  }
  alpaka::memset(queue, *m_nclusters, 0u);

  detail::computeNearestHighers<internal::Acc>(queue,
                                               work_division,
                                               m_tiles->view(),
                                               dev_points.view(),
                                               m_outlier_distance,
                                               m_seeding_distance,
                                               m_min_density,
                                               metric,
											   m_nclusters->data());
  
  detail::assignPointsToClusters<internal::Acc>(
      queue, block_size, dev_points.view());

  // optional step, skipped entirely when disabled so that the results of plain CLUE are unchanged
  if (m_merge_threshold.has_value()) {
    detail::mergeClusters<internal::Acc>(queue,
                                         block_size,
                                         m_tiles->view(),
                                         dev_points.view(),
                                         m_seeds.value(),
                                         *m_merge_threshold,
                                         m_density_radius,
                                         metric,
                                         m_allocator);
  }

  alpaka::wait(queue);
  internal::points_interface<std::remove_cvref_t<decltype(dev_points)>>::mark_clustered(dev_points);
}

template <std::size_t Ndim, std::floating_point DataType, concepts::allocator AllocatorType>
template <std::floating_point InputType,
          concepts::convolutional_kernel Kernel,
          concepts::distance_metric<Ndim> DistanceMetric>
void Clusterer<Ndim, DataType, AllocatorType>::make_clusters_batched(
    clue::PointsDevice<Ndim, InputType>& dev_points,
    std::span<const uint32_t> batch_item_sizes,
    const DistanceMetric& metric,
    const Kernel& kernel,
    Queue& queue) {
  if (m_merge_threshold.has_value()) {
    // TODO: the saddle kernel and the union-find must be run per batch item
    throw std::logic_error(
        "The merging of the clusters is not supported yet for batched clustering");
  }
  constexpr std::size_t block_size = 256;
  const auto batch_size = batch_item_sizes.size();

  const auto max_event_size =
      std::reduce(batch_item_sizes.begin(), batch_item_sizes.end(), 0u, nostd::maximum<uint32_t>{});

  auto event_offsets = alpaka::allocMappedBuf<std::size_t>(host, platform<Queue>(), batch_size + 1);
  event_offsets[0] = 0;
  std::inclusive_scan(batch_item_sizes.begin(), batch_item_sizes.end(), event_offsets.data() + 1);
  if (event_offsets[batch_size] != static_cast<std::size_t>(dev_points.size())) {
    throw std::invalid_argument(
        "The sum of the elements in the batches must be equal to the total number of points");
  }

  m_event_offsets.emplace(
      internal::alloc_device_buffer<std::size_t>(queue, batch_size + 1, m_allocator));
  m_batch_size = batch_size;
  auto& d_event_offsets = *m_event_offsets;
  alpaka::memcpy(queue, d_event_offsets, event_offsets);

  m_tiles->template fill_batch<internal::Acc>(
      queue, dev_points, d_event_offsets, max_event_size, m_allocator);

  detail::computeLocalDensityBatched<internal::Acc2D>(queue,
                                                      m_tiles->view(),
                                                      dev_points.view(),
                                                      kernel,
                                                      m_density_radius,
                                                      metric,
                                                      d_event_offsets,
                                                      max_event_size,
                                                      block_size);

  if (!m_nclusters.has_value()) {
    m_nclusters.emplace(internal::alloc_device_buffer<std::size_t>(queue, Scalar{}, m_allocator));
  }
  alpaka::memset(queue, *m_nclusters, 0u);

  detail::computeNearestHighersBatched<internal::Acc2D>(queue,
                                                        m_tiles->view(),
                                                        dev_points.view(),
                                                        m_outlier_distance,
                                                        m_seeding_distance,
                                                        m_min_density,
                                                        metric,
                                                        m_nclusters->data(),
                                                        d_event_offsets,
                                                        max_event_size,
                                                        block_size);

  detail::assignPointsToClusters<internal::Acc>(queue, block_size, dev_points.view());

  alpaka::wait(queue);
  internal::points_interface<std::remove_cvref_t<decltype(dev_points)>>::mark_clustered(dev_points);
}

CLUE_NAMESPACE_END
