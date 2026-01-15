
#pragma once

#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/detail/SetupFollowers.hpp"
#include "CLUEstering/core/detail/SetupSeeds.hpp"
#include "CLUEstering/core/detail/SetupTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/utils/get_clusters.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>

namespace clue {

  template <std::size_t Ndim>
  Clusterer<Ndim>::Clusterer(
      float dc, float rhoc, std::optional<float> dm, std::optional<float> seed_dc, int pPBin)
      : m_dc{dc},
        m_seed_dc{seed_dc.value_or(dc)},
        m_rhoc{rhoc},
        m_dm{dm.value_or(dc)},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if (m_dc <= 0.f || m_rhoc < 0.f || m_dm <= 0.f || m_seed_dc <= 0.f || m_pointsPerTile <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
  }

  template <std::size_t Ndim>
  inline Clusterer<Ndim>::Clusterer(Queue&,
                                    float dc,
                                    float rhoc,
                                    std::optional<float> dm,
                                    std::optional<float> seed_dc,
                                    int pPBin)
      : m_dc{dc},
        m_seed_dc{seed_dc.value_or(dc)},
        m_rhoc{rhoc},
        m_dm{dm.value_or(dc)},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if (m_dc <= 0.f || m_rhoc < 0.f || m_dm <= 0.f || m_seed_dc <= 0.f || m_pointsPerTile <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
  }

  template <std::size_t Ndim>
  void Clusterer<Ndim>::setParameters(
      float dc, float rhoc, std::optional<float> dm, std::optional<float> seed_dc, int pPBin) {
    m_dc = dc;
    m_dm = dm.value_or(dc);
    m_seed_dc = seed_dc.value_or(dc);
    m_rhoc = rhoc;
    m_pointsPerTile = pPBin;

    if (m_dc <= 0.f || m_rhoc < 0.f || m_dm <= 0.f || m_seed_dc <= 0.f || m_pointsPerTile <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
  }

  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsHost& h_points,
                                             const DistanceMetric& metric,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    auto d_points = PointsDevice(queue, h_points.size());

    setup(queue, h_points, d_points);
    make_clusters_impl(h_points, d_points, metric, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<Ndim>::make_clusters(PointsHost& h_points,
                                             const DistanceMetric& metric,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    auto device = alpaka::getDevByIdx(Platform{}, 0u);
    Queue queue(device);
    auto d_points = PointsDevice(queue, h_points.size());

    setup(queue, h_points, d_points);
    make_clusters_impl(h_points, d_points, metric, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsHost& h_points,
                                             PointsDevice& dev_points,
                                             const DistanceMetric& metric,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    setup(queue, h_points, dev_points);
    make_clusters_impl(h_points, dev_points, metric, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsDevice& dev_points,
                                             const DistanceMetric& metric,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    detail::setup_tiles(queue, m_tiles, dev_points, m_pointsPerTile, m_wrappedCoordinates);
    detail::setup_followers(queue, m_followers, dev_points.size());
    make_clusters_impl(dev_points, metric, kernel, queue, block_size);
    alpaka::wait(queue);
  }

  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsHost& h_points,
                                             PointsDevice& dev_points,
                                             std::span<const uint32_t> batch_item_sizes,
                                             const DistanceMetric& metric,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    const auto batch_size = batch_item_sizes.size();
    setup_batch(queue, h_points, dev_points, batch_size);

    const auto max_event_size = std::reduce(
        batch_item_sizes.begin(), batch_item_sizes.end(), 0u, nostd::maximum<uint32_t>{});

    auto event_offsets = clue::make_host_buffer<std::size_t[]>(batch_size + 1);
    event_offsets[0] = 0;
    std::inclusive_scan(batch_item_sizes.begin(), batch_item_sizes.end(), event_offsets.data() + 1);
    auto d_event_offsets = clue::make_device_buffer<std::size_t[]>(queue, batch_size + 1);
    alpaka::memcpy(queue, d_event_offsets, event_offsets);
    alpaka::wait(queue);

    const auto n_points = h_points.size();
    m_tiles->template fill_batch<Acc>(queue, dev_points, n_points, d_event_offsets, max_event_size);

    detail::computeLocalDensityBatched<internal::Acc2D>(queue,
                                                        m_tiles->view(),
                                                        dev_points.view(),
                                                        kernel,
                                                        m_dc,
                                                        metric,
                                                        d_event_offsets,
                                                        max_event_size,
                                                        block_size);
    auto seed_candidates = 0ul;
    detail::computeNearestHighersBatched<internal::Acc2D>(queue,
                                                          m_tiles->view(),
                                                          dev_points.view(),
                                                          m_dm,
                                                          metric,
                                                          seed_candidates,
                                                          d_event_offsets,
                                                          max_event_size,
                                                          block_size);
    detail::setup_seeds(queue, m_seeds, seed_candidates);
    m_event_associations = make_device_buffer<std::int32_t[]>(queue, seed_candidates);

    detail::findClusterSeedsBatched<internal::Acc2D>(queue,
                                                     m_seeds.value(),
                                                     dev_points.view(),
                                                     m_seed_dc,
                                                     metric,
                                                     m_rhoc,
                                                     d_event_offsets,
                                                     max_event_size,
                                                     m_event_associations->data(),
                                                     block_size);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds.value(), m_followers->view(), dev_points.view());

    clue::copyToHost(queue, h_points, dev_points);
    h_points.mark_clustered();
    dev_points.mark_clustered();
  }

  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsDevice& dev_points,
                                             std::span<const uint32_t> batch_item_sizes,
                                             const DistanceMetric& metric,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    const auto batch_size = batch_item_sizes.size();
    setup_batch(queue, dev_points, batch_size);

    const auto max_event_size = std::reduce(
        batch_item_sizes.begin(), batch_item_sizes.end(), 0u, nostd::maximum<uint32_t>{});

    auto event_offsets = clue::make_host_buffer<std::size_t[]>(batch_size + 1);
    event_offsets[0] = 0;
    std::inclusive_scan(batch_item_sizes.begin(), batch_item_sizes.end(), event_offsets.data() + 1);
    auto d_event_offsets = clue::make_device_buffer<std::size_t[]>(queue, batch_size + 1);
    alpaka::memcpy(queue, d_event_offsets, event_offsets);
    alpaka::wait(queue);

    const auto n_points = dev_points.size();
    m_tiles->template fill_batch<Acc>(queue, dev_points, n_points, d_event_offsets, max_event_size);

    detail::computeLocalDensityBatched<internal::Acc2D>(queue,
                                                        m_tiles->view(),
                                                        dev_points.view(),
                                                        kernel,
                                                        m_dc,
                                                        metric,
                                                        d_event_offsets,
                                                        max_event_size,
                                                        block_size);
    auto seed_candidates = 0ul;
    detail::computeNearestHighersBatched<internal::Acc2D>(queue,
                                                          m_tiles->view(),
                                                          dev_points.view(),
                                                          m_dm,
                                                          metric,
                                                          seed_candidates,
                                                          d_event_offsets,
                                                          max_event_size,
                                                          block_size);
    detail::setup_seeds(queue, m_seeds, seed_candidates);
    m_event_associations = make_device_buffer<std::int32_t[]>(queue, seed_candidates);

    detail::findClusterSeedsBatched<internal::Acc2D>(queue,
                                                     m_seeds.value(),
                                                     dev_points.view(),
                                                     m_seed_dc,
                                                     metric,
                                                     m_rhoc,
                                                     d_event_offsets,
                                                     max_event_size,
                                                     m_event_associations->data(),
                                                     block_size);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds.value(), m_followers->view(), dev_points.view());

    dev_points.mark_clustered();
  }

  template <std::size_t Ndim>
  template <std::ranges::contiguous_range TRange>
    requires std::integral<std::ranges::range_value_t<TRange>>
  inline void Clusterer<Ndim>::setWrappedCoordinates(const TRange& wrapped_coordinates) {
    std::ranges::copy(wrapped_coordinates, m_wrappedCoordinates.begin());
  }
  template <std::size_t Ndim>
  template <std::integral... TArgs>
  inline void Clusterer<Ndim>::setWrappedCoordinates(TArgs... wrappedCoordinates) {
    m_wrappedCoordinates = {static_cast<uint8_t>(wrappedCoordinates)...};
  }

  template <std::size_t Ndim>
  inline host_associator Clusterer<Ndim>::getClusters(const PointsHost& h_points) {
    return clue::get_clusters(h_points);
  }

  template <std::size_t Ndim>
  inline AssociationMap<Device> Clusterer<Ndim>::getClusters(Queue& queue,
                                                             const PointsDevice& d_points) {
    return clue::get_clusters(queue, d_points);
  }

  template <std::size_t Ndim>
  inline host_associator Clusterer<Ndim>::getSampleAssociations(Queue& queue,
                                                                const PointsHost& h_points) {
    auto event_associations = make_host_buffer<std::int32_t[]>(h_points.n_clusters());
    alpaka::memcpy(queue, event_associations, m_event_associations->data());
    alpaka::wait(queue);
    return internal::make_associator(event_associations, h_points.n_clusters());
  }

  template <std::size_t Ndim>
  inline AssociationMap<Device> Clusterer<Ndim>::getSampleAssociations(
      Queue& queue, const PointsDevice& d_points) {
    internal::make_associator(queue, m_event_associations->data(), d_points.n_clusters());
  }

  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  void Clusterer<Ndim>::make_clusters_impl(PointsHost& h_points,
                                           PointsDevice& dev_points,
                                           const DistanceMetric& metric,
                                           const Kernel& kernel,
                                           Queue& queue,
                                           std::size_t block_size) {
    const auto n_points = h_points.size();
    m_tiles->template fill<Acc>(queue, dev_points, n_points);

    const Idx grid_size = clue::divide_up_by(n_points, block_size);
    auto work_division = clue::make_workdiv<Acc>(grid_size, block_size);

    detail::computeLocalDensity<Acc>(
        queue, work_division, m_tiles->view(), dev_points.view(), kernel, m_dc, metric, n_points);
    auto seed_candidates = 0ul;
    detail::computeNearestHighers<Acc>(queue,
                                       work_division,
                                       m_tiles->view(),
                                       dev_points.view(),
                                       m_dm,
                                       metric,
                                       seed_candidates,
                                       n_points);
    detail::setup_seeds(queue, m_seeds, seed_candidates);
    detail::findClusterSeeds<Acc>(queue,
                                  work_division,
                                  m_seeds.value(),
                                  dev_points.view(),
                                  m_seed_dc,
                                  metric,
                                  m_rhoc,
                                  n_points);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds.value(), m_followers->view(), dev_points.view());

    clue::copyToHost(queue, h_points, dev_points);
    h_points.mark_clustered();
    dev_points.mark_clustered();
  }

  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  void Clusterer<Ndim>::make_clusters_impl(PointsDevice& dev_points,
                                           const DistanceMetric& metric,
                                           const Kernel& kernel,
                                           Queue& queue,
                                           std::size_t block_size) {
    const auto n_points = dev_points.size();
    m_tiles->template fill<Acc>(queue, dev_points, n_points);

    const Idx grid_size = clue::divide_up_by(n_points, block_size);
    auto work_division = clue::make_workdiv<Acc>(grid_size, block_size);

    detail::computeLocalDensity<Acc>(
        queue, work_division, m_tiles->view(), dev_points.view(), kernel, m_dc, metric, n_points);
    auto seed_candidates = 0ul;
    detail::computeNearestHighers<Acc>(queue,
                                       work_division,
                                       m_tiles->view(),
                                       dev_points.view(),
                                       m_dm,
                                       metric,
                                       seed_candidates,
                                       n_points);
    detail::setup_seeds(queue, m_seeds, seed_candidates);
    detail::findClusterSeeds<Acc>(queue,
                                  work_division,
                                  m_seeds.value(),
                                  dev_points.view(),
                                  m_seed_dc,
                                  metric,
                                  m_rhoc,
                                  n_points);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds.value(), m_followers->view(), dev_points.view());

    alpaka::wait(queue);
    dev_points.mark_clustered();
  }

}  // namespace clue
