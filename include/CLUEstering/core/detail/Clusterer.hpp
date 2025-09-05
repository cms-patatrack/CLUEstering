
#pragma once

#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/CLUEAlpakaKernels.hpp"
#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/Tiles.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <execution>
#include <ranges>
#include <vector>

namespace clue {

  template <uint8_t Ndim>
  inline Clusterer<Ndim>::Clusterer(float dc, float rhoc, float dm, float seed_dc, int pPBin)
      : m_dc{dc},
        m_seed_dc{seed_dc},
        m_rhoc{rhoc},
        m_dm{dm},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if (seed_dc < 0.f) {
      m_seed_dc = dc;
    }
  }

  template <uint8_t Ndim>
  inline Clusterer<Ndim>::Clusterer(
      Queue& queue, float dc, float rhoc, float dm, float seed_dc, int pPBin)
      : m_dc{dc},
        m_seed_dc{seed_dc},
        m_rhoc{rhoc},
        m_dm{dm},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if (seed_dc < 0.f) {
      m_seed_dc = dc;
    }
    init_device(queue);
  }

  template <uint8_t Ndim>
  inline Clusterer<Ndim>::Clusterer(Queue& queue,
                                    TilesDevice* tile_buffer,
                                    float dc,
                                    float rhoc,
                                    float dm,
                                    float seed_dc,
                                    int pPBin)
      : m_dc{dc},
        m_seed_dc{seed_dc},
        m_rhoc{rhoc},
        m_dm{dm},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if (seed_dc < 0.f) {
      m_seed_dc = dc;
    }
    init_device(queue, tile_buffer);
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setParameters(float dc, float rhoc, float dm, float seed_dc, int pPBin) {
    m_dc = dc;
    m_seed_dc = seed_dc < 0.f ? dc : seed_dc;
    m_rhoc = rhoc;
    m_dm = dm;
    m_pointsPerTile = pPBin;
  }

  template <uint8_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsHost& h_points,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    d_points = std::make_optional<PointsDevice>(queue, h_points.size());
    auto& dev_points = *d_points;

    setup(queue, h_points, dev_points);
    make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <uint8_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(PointsHost& h_points,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    auto device = alpaka::getDevByIdx(Platform{}, 0u);
    Queue queue(device);
    init_device(queue);

    d_points = std::make_optional<PointsDevice>(queue, h_points.size());
    auto& dev_points = *d_points;

    setup(queue, h_points, dev_points);
    make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <uint8_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsHost& h_points,
                                             PointsDevice& dev_points,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    setup(queue, h_points, dev_points);
    make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <uint8_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(PointsHost& h_points,
                                             PointsDevice& dev_points,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    auto device = alpaka::getDevByIdx(Platform{}, 0u);
    Queue queue(device);
    init_device(queue);

    setup(queue, h_points, dev_points);
    make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <uint8_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsDevice& dev_points,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    setupTiles(queue, dev_points);
    setupFollowers(queue, dev_points.size());
    alpaka::memset(queue, *m_seeds, 0x00);
    make_clusters_impl(dev_points, kernel, queue, block_size);
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  inline void Clusterer<Ndim>::setWrappedCoordinates(
      const std::array<uint8_t, Ndim>& wrappedCoordinates) {
    m_wrappedCoordinates = wrappedCoordinates;
  }
  template <uint8_t Ndim>
  inline void Clusterer<Ndim>::setWrappedCoordinates(
      std::array<uint8_t, Ndim>&& wrappedCoordinates) {
    m_wrappedCoordinates = std::move(wrappedCoordinates);
  }
  template <uint8_t Ndim>
  template <typename... TArgs>
  inline void Clusterer<Ndim>::setWrappedCoordinates(TArgs... wrappedCoordinates) {
    m_wrappedCoordinates = {wrappedCoordinates...};
  }

  template <uint8_t Ndim>
  inline std::vector<std::vector<int>> Clusterer<Ndim>::getClusters(const PointsHost& h_points) {
    return clue::compute_clusters_points(h_points.clusterIndexes());
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::init_device(Queue& queue) {
    m_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue);
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::init_device(Queue& queue, TilesDevice* tile_buffer) {
    m_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue);

    // load tiles from outside
    m_tiles = *tile_buffer;
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupTiles(Queue& queue, const PointsHost& h_points) {
    // TODO: reconsider the way that we compute the number of tiles
    auto nTiles =
        static_cast<int32_t>(std::ceil(h_points.size() / static_cast<float>(m_pointsPerTile)));
    const auto nPerDim = static_cast<int32_t>(std::ceil(std::pow(nTiles, 1. / Ndim)));
    nTiles = static_cast<int32_t>(std::pow(nPerDim, Ndim));

    if (!m_tiles.has_value()) {
      m_tiles = std::make_optional<TilesDevice>(queue, h_points.size(), nTiles);
    }
    // check if tiles are large enough for current data
    if (!(m_tiles->extents().values >= static_cast<std::size_t>(h_points.size())) or
        !(m_tiles->extents().keys >= static_cast<std::size_t>(nTiles))) {
      m_tiles->initialize(h_points.size(), nTiles, nPerDim, queue);
    } else {
      m_tiles->reset(h_points.size(), nTiles, nPerDim, queue);
    }

    auto min_max = clue::make_host_buffer<CoordinateExtremes>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);
    detail::compute_tile_size(min_max.data(), tile_sizes.data(), h_points, nPerDim);

    alpaka::memcpy(queue, m_tiles->minMax(), min_max);
    alpaka::memcpy(queue, m_tiles->tileSize(), tile_sizes);
    alpaka::memcpy(
        queue, m_tiles->wrapped(), clue::make_host_view(m_wrappedCoordinates.data(), Ndim));
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupTiles(Queue& queue, const PointsDevice& d_points) {
    auto nTiles =
        static_cast<int32_t>(std::ceil(d_points.size() / static_cast<float>(m_pointsPerTile)));
    const auto nPerDim = static_cast<int32_t>(std::ceil(std::pow(nTiles, 1. / Ndim)));
    nTiles = static_cast<int32_t>(std::pow(nPerDim, Ndim));

    if (!m_tiles.has_value()) {
      m_tiles = std::make_optional<TilesDevice>(queue, d_points.size(), nTiles);
    }
    // check if tiles are large enough for current data
    if (!(m_tiles->extents().values >= static_cast<std::size_t>(d_points.size())) or
        !(m_tiles->extents().keys >= static_cast<std::size_t>(nTiles))) {
      m_tiles->initialize(d_points.size(), nTiles, nPerDim, queue);
    } else {
      m_tiles->reset(d_points.size(), nTiles, nPerDim, queue);
    }

    auto min_max = clue::make_host_buffer<CoordinateExtremes>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);
    detail::compute_tile_size(queue, min_max.data(), tile_sizes.data(), d_points, nPerDim);

    alpaka::memcpy(queue, m_tiles->minMax(), min_max);
    alpaka::memcpy(queue, m_tiles->tileSize(), tile_sizes);
    alpaka::memcpy(
        queue, m_tiles->wrapped(), clue::make_host_view(m_wrappedCoordinates.data(), Ndim));
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupFollowers(Queue& queue, int32_t n_points) {
    if (!m_followers.has_value()) {
      m_followers = std::make_optional<FollowersDevice>(n_points, queue);
    }

    if (!(m_followers->extents() >= n_points)) {
      m_followers->initialize(n_points, queue);
    } else {
      m_followers->reset(n_points, queue);
    }
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupPoints(const PointsHost& h_points,
                                    PointsDevice& dev_points,
                                    Queue& queue) {
    clue::copyToDevice(queue, dev_points, h_points);
    alpaka::memset(queue, *m_seeds, 0x00);
  }

  template <uint8_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  void Clusterer<Ndim>::make_clusters_impl(PointsHost& h_points,
                                           PointsDevice& dev_points,
                                           const Kernel& kernel,
                                           Queue& queue,
                                           std::size_t block_size) {
    const auto n_points = h_points.size();
    m_tiles->template fill<Acc>(queue, dev_points, n_points);

    const Idx grid_size = clue::divide_up_by(n_points, block_size);
    auto work_division = clue::make_workdiv<Acc>(grid_size, block_size);

    detail::computeLocalDensity<Acc>(
        queue, work_division, m_tiles->view(), dev_points.view(), kernel, m_dc, n_points);
    detail::computeNearestHighers<Acc>(
        queue, work_division, m_tiles->view(), dev_points.view(), m_dm, n_points);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::findClusterSeeds<Acc>(
        queue, work_division, m_seeds->data(), dev_points.view(), m_seed_dc, m_rhoc, n_points);
    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds->data(), m_followers->view(), dev_points.view());

    clue::copyToHost(queue, h_points, dev_points);
  }

  template <uint8_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  void Clusterer<Ndim>::make_clusters_impl(PointsDevice& dev_points,
                                           const Kernel& kernel,
                                           Queue& queue,
                                           std::size_t block_size) {
    const auto n_points = dev_points.size();
    m_tiles->template fill<Acc>(queue, dev_points, n_points);

    const Idx grid_size = clue::divide_up_by(n_points, block_size);
    auto work_division = clue::make_workdiv<Acc>(grid_size, block_size);

    detail::computeLocalDensity<Acc>(
        queue, work_division, m_tiles->view(), dev_points.view(), kernel, m_dc, n_points);
    detail::computeNearestHighers<Acc>(
        queue, work_division, m_tiles->view(), dev_points.view(), m_dm, n_points);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::findClusterSeeds<Acc>(
        queue, work_division, m_seeds->data(), dev_points.view(), m_seed_dc, m_rhoc, n_points);
    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds->data(), m_followers->view(), dev_points.view());

    alpaka::wait(queue);
  }

}  // namespace clue
