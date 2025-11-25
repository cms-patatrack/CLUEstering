
#pragma once

#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/core/DistanceParameter.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/utils/get_clusters.hpp"

#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <concepts>
#include <cstdint>
#include <ranges>

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
  inline Clusterer<Ndim>::Clusterer(Queue& queue,
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
    init_device(queue);
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
                                             std::size_t block_size) {
    auto d_points = PointsDevice(queue, h_points.size());

    setup(queue, h_points, d_points);
    make_clusters_impl(h_points, d_points, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(PointsHost& h_points,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    auto device = alpaka::getDevByIdx(Platform{}, 0u);
    Queue queue(device);
    init_device(queue);

    auto d_points = PointsDevice(queue, h_points.size());

    setup(queue, h_points, d_points);
    make_clusters_impl(h_points, d_points, kernel, queue, block_size);
    alpaka::wait(queue);
  }
  template <std::size_t Ndim>
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
  template <std::size_t Ndim>
  template <concepts::convolutional_kernel Kernel>
  inline void Clusterer<Ndim>::make_clusters(Queue& queue,
                                             PointsDevice& dev_points,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    detail::setup_tiles(queue, m_tiles, dev_points, m_pointsPerTile, m_wrappedCoordinates);
    detail::setup_followers(queue, m_followers, dev_points.size());
    alpaka::memset(queue, *m_seeds, 0x00);
    make_clusters_impl(dev_points, kernel, queue, block_size);
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
  void Clusterer<Ndim>::init_device(Queue& queue) {
    m_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue);
  }

  template <std::size_t Ndim>
  void Clusterer<Ndim>::init_device(Queue& queue, TilesDevice* tile_buffer) {
    m_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue);

    // load tiles from outside
    m_tiles = *tile_buffer;
  }

  template <std::size_t Ndim>
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
    detail::findClusterSeeds<Acc>(queue,
                                  work_division,
                                  m_seeds->data(),
                                  m_tiles->view(),
                                  dev_points.view(),
                                  m_seed_dc,
                                  m_rhoc,
                                  n_points);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds->data(), m_followers->view(), dev_points.view());

    clue::copyToHost(queue, h_points, dev_points);
    h_points.mark_clustered();
    dev_points.mark_clustered();
  }

  template <std::size_t Ndim>
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
    detail::findClusterSeeds<Acc>(queue,
                                  work_division,
                                  m_seeds->data(),
                                  m_tiles->view(),
                                  dev_points.view(),
                                  m_seed_dc,
                                  m_rhoc,
                                  n_points);

    m_followers->template fill<Acc>(queue, dev_points);

    detail::assignPointsToClusters<Acc>(
        queue, block_size, m_seeds->data(), m_followers->view(), dev_points.view());

    alpaka::wait(queue);
    dev_points.mark_clustered();
  }

}  // namespace clue
