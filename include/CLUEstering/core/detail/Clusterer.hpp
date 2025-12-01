
#pragma once

#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/core/DistanceParameter.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/core/detail/ComputeGridSize.hpp"
#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/core/detail/ComputeBatches.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/detail/SetupFollowers.hpp"
#include "CLUEstering/core/detail/SetupSeeds.hpp"
#include "CLUEstering/core/detail/SetupTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/utils/get_clusters.hpp"

#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>

namespace clue {

  template <std::size_t Ndim>
  Clusterer<Ndim>::Clusterer(DistanceParameter<Ndim> dc,
                             float rhoc,
                             DistanceParameter<Ndim> dm,
                             DistanceParameter<Ndim> seed_dc,
                             int pPBin)
      : m_dc{std::move(dc)},
        m_seed_dc{std::move(seed_dc)},
        m_rhoc{rhoc},
        m_dm{std::move(dm)},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if (dc <= 0.f || rhoc <= 0.f || pPBin <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
    if (dm <= 0.f) {
      m_dm = dc;
    }
    if (seed_dc < 0.f) {
      m_seed_dc = dc;
    }
  }

  template <std::size_t Ndim>
  inline Clusterer<Ndim>::Clusterer(Queue&,
                                    DistanceParameter<Ndim> dc,
                                    float rhoc,
                                    DistanceParameter<Ndim> dm,
                                    DistanceParameter<Ndim> seed_dc,
                                    int pPBin)
      : m_dc{dc},
        m_seed_dc{seed_dc},
        m_rhoc{rhoc},
        m_dm{dm},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if (dc <= 0.f || rhoc <= 0.f || pPBin <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
    if (dm <= 0.f) {
      m_dm = dc;
    }
    if (seed_dc < 0.f) {
      m_seed_dc = dc;
    }
  }

  template <std::size_t Ndim>
  void Clusterer<Ndim>::setParameters(DistanceParameter<Ndim> dc,
                                      float rhoc,
                                      DistanceParameter<Ndim> dm,
                                      DistanceParameter<Ndim> seed_dc,
                                      int pPBin) {
    if (dc <= 0.f || rhoc < 0.f || pPBin <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
    m_dc = dc;
    m_dm = dm < 0.f ? dc : dm;
    m_seed_dc = seed_dc < 0.f ? dc : seed_dc;
    m_rhoc = rhoc;
    m_dm = dm < 0.f ? dc : dm;
    m_pointsPerTile = pPBin;
  }

  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsHost& h_points,
                                             const Kernel& kernel,
                                             std::span<std::size_t> batch_event_sizes,
                                             std::size_t block_size) {
    auto d_points = PointsDevice(queue, h_points.size());

    const auto batches = batch_event_sizes.empty() ? 1ul : batch_event_sizes.size();
    setup(queue, h_points, d_points, batch_event_sizes);
    make_clusters_impl(h_points, d_points, kernel, queue, batch_event_sizes, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(PointsHost& h_points,
                                             const Kernel& kernel,
                                             std::span<std::size_t> batch_event_sizes,
                                             std::size_t block_size) {
    const auto device = alpaka::getDevByIdx(Platform{}, 0u);
    Queue queue(device);
    auto d_points = PointsDevice(queue, h_points.size());

    const auto batches = batch_event_sizes.empty() ? 1ul : batch_event_sizes.size();

    setup(queue, h_points, d_points, batch_event_sizes);
    make_clusters_impl(h_points, d_points, kernel, queue, batch_event_sizes, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsHost& h_points,
                                             PointsDevice& dev_points,
                                             const Kernel& kernel,
                                             std::span<std::size_t> batch_event_sizes,
                                             std::size_t block_size) {
    const auto batches = batch_event_sizes.empty() ? 1ul : batch_event_sizes.size();
    setup(queue, h_points, dev_points, batch_event_sizes);
    make_clusters_impl(h_points, dev_points, kernel, queue, batch_event_sizes, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsDevice& dev_points,
                                             const Kernel& kernel,
                                             std::span<std::size_t> batch_event_sizes,
                                             std::size_t block_size) {
    const auto batches = batch_event_sizes.empty() ? 1ul : batch_event_sizes.size();
    detail::setup_tiles(queue, m_tiles, dev_points, batches, m_pointsPerTile, m_wrappedCoordinates);
    detail::setup_followers(queue, m_followers, dev_points.size());
    make_clusters_impl(dev_points, kernel, queue, batch_event_sizes, block_size);
    alpaka::wait(queue);
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
  template <concepts::convolutional_kernel Kernel>
  void Clusterer<Ndim>::make_clusters_impl(PointsHost& h_points,
                                           PointsDevice& dev_points,
                                           const Kernel& kernel,
                                           Queue& queue,
                                           std::span<std::size_t> batch_event_sizes,
                                           std::size_t block_size) {
    const auto n_points = h_points.size();
    m_tiles->template fill<internal::Acc2D>(queue, dev_points, n_points);

    auto max_batch_item_size = std::reduce(
        batch_event_sizes.begin(), batch_event_sizes.end(), 0ul, nostd::maximum<std::size_t>{});

    auto h_cumulative_batch_item_sizes =
        clue::make_host_buffer<std::size_t[]>(queue, batch_event_sizes.size());
    std::inclusive_scan(
        batch_event_sizes.begin(), batch_event_sizes.end(), h_cumulative_batch_item_sizes.data());
    auto d_cumulative_batch_item_sizes =
        clue::make_device_buffer<std::size_t[]>(queue, batch_event_sizes.size());
    alpaka::memcpy(queue, d_cumulative_batch_item_sizes, h_cumulative_batch_item_sizes);

    const auto batch_grid_size = clue::divide_up_by(n_points, block_size);
    const auto blocks_per_batch = detail::compute_batch_blocks(max_batch_item_size, block_size);
    auto batch_work_division = clue::make_workdiv<internal::Acc2D>(
        {batch_grid_size, batch_grid_size}, {block_size, blocks_per_batch});

    detail::computeLocalDensity<internal::Acc2D>(queue,
                                                 batch_work_division,
                                                 m_tiles->view(),
                                                 dev_points.view(),
                                                 kernel,
                                                 m_dc,
                                                 n_points,
                                                 d_cumulative_batch_item_sizes);
    auto seed_candidates = 0ul;
    detail::computeNearestHighers<internal::Acc2D>(queue,
                                                   batch_work_division,
                                                   m_tiles->view(),
                                                   dev_points.view(),
                                                   m_dm,
                                                   seed_candidates,
                                                   n_points,
                                                   d_cumulative_batch_item_sizes);
    detail::setup_seeds(queue, m_seeds, seed_candidates);
    const auto grid_size = clue::divide_up_by(n_points, block_size);
    auto work_division = clue::make_workdiv<Acc>(grid_size, block_size);
    detail::findClusterSeeds<Acc>(queue,
                                  work_division,
                                  m_seeds.value(),
                                  m_tiles->view(),
                                  dev_points.view(),
                                  m_seed_dc,
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
  template <concepts::convolutional_kernel Kernel>
  void Clusterer<Ndim>::make_clusters_impl(PointsDevice& dev_points,
                                           const Kernel& kernel,
                                           Queue& queue,
                                           std::span<std::size_t> batch_event_sizes,
                                           std::size_t block_size) {
    const auto n_points = dev_points.size();
    m_tiles->template fill<internal::Acc2D>(queue, dev_points, n_points);

    auto max_batch_item_size = std::reduce(
        batch_event_sizes.begin(), batch_event_sizes.end(), 0ul, nostd::maximum<std::size_t>{});

    auto h_cumulative_batch_item_sizes =
        clue::make_host_buffer<std::size_t[]>(queue, batch_event_sizes.size());
    std::inclusive_scan(
        batch_event_sizes.begin(), batch_event_sizes.end(), h_cumulative_batch_item_sizes.data());
    auto d_cumulative_batch_item_sizes =
        clue::make_device_buffer<std::size_t[]>(queue, batch_event_sizes.size());
    alpaka::memcpy(queue, d_cumulative_batch_item_sizes, h_cumulative_batch_item_sizes);

    const auto batch_grid_size = clue::divide_up_by(n_points, block_size);
    const auto blocks_per_batch = detail::compute_batch_blocks(max_batch_item_size, block_size);
    auto batch_work_division = clue::make_workdiv<internal::Acc2D>(
        {batch_grid_size, batch_grid_size}, {block_size, blocks_per_batch});

    detail::computeLocalDensity<internal::Acc2D>(queue,
                                                 batch_work_division,
                                                 m_tiles->view(),
                                                 dev_points.view(),
                                                 kernel,
                                                 m_dc,
                                                 n_points,
                                                 d_cumulative_batch_item_sizes);
    auto seed_candidates = 0ul;
    detail::computeNearestHighers<internal::Acc2D>(queue,
                                                   batch_work_division,
                                                   m_tiles->view(),
                                                   dev_points.view(),
                                                   m_dm,
                                                   seed_candidates,
                                                   n_points,
                                                   d_cumulative_batch_item_sizes);
    detail::setup_seeds(queue, m_seeds, seed_candidates);
    const auto grid_size = clue::divide_up_by(n_points, block_size);
    auto work_division = clue::make_workdiv<Acc>(grid_size, block_size);
    detail::findClusterSeeds<Acc>(queue,
                                  work_division,
                                  m_seeds.value(),
                                  m_tiles->view(),
                                  dev_points.view(),
                                  m_seed_dc,
                                  m_rhoc,
                                  n_points);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds.value(), m_followers->view(), dev_points.view());

    alpaka::wait(queue);
    dev_points.mark_clustered();
  }

}  // namespace clue
