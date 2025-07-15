
#pragma once

#include "CLUEstering/DataFormats/PointsHost.hpp"
#include "CLUEstering/DataFormats/PointsDevice.hpp"
#include "CLUEstering/DataFormats/alpaka/TilesAlpaka.hpp"
#include "CLUEstering/DataFormats/alpaka/Followers.hpp"
#include "CLUEstering/CLUE/CLUEAlpakaKernels.hpp"
#include "CLUEstering/CLUE/ConvolutionalKernel.hpp"
#include "CLUEstering/utility/validation.hpp"

#include <algorithm>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

namespace clue {

  template <uint8_t Ndim>
  class Clusterer {
  public:
    using Device = ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Device;
    using Queue = ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Queue;
    using Acc1D = ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Acc1D;
    using Platform = ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Platform;
    using KernelCalculateLocalDensity =
        ALPAKA_ACCELERATOR_NAMESPACE_CLUE::KernelCalculateLocalDensity;
    using KernelCalculateNearestHigher =
        ALPAKA_ACCELERATOR_NAMESPACE_CLUE::KernelCalculateNearestHigher;
    using KernelFindClusters = ALPAKA_ACCELERATOR_NAMESPACE_CLUE::KernelFindClusters;
    using KernelAssignClusters = ALPAKA_ACCELERATOR_NAMESPACE_CLUE::KernelAssignClusters;

    using CoordinateExtremes = clue::CoordinateExtremes<Ndim>;
    using PointsHost = clue::PointsHost<Ndim>;
    using PointsDevice = clue::PointsDevice<Ndim, Device>;
    using TilesDevice = clue::TilesAlpaka<Ndim, Device>;
    using FollowersDevice = clue::Followers<Device>;

    inline static constexpr auto reserve = ALPAKA_ACCELERATOR_NAMESPACE_CLUE::reserve;

    explicit Clusterer(float dc, float rhoc, float dm, float seed_dc = -1.f, int pPBin = 128)
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
    explicit Clusterer(
        Queue& queue, float dc, float rhoc, float dm, float seed_dc = -1.f, int pPBin = 128)
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
    explicit Clusterer(Queue& queue,
                       TilesDevice* tile_buffer,
                       float dc,
                       float rhoc,
                       float dm,
                       float seed_dc = -1.f,
                       int pPBin = 128)
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

    TilesAlpakaView<Ndim>* m_tiles;
    VecArray<int32_t, reserve>* m_seeds;
    FollowersView* m_followers;

    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       const KernelType& kernel,
                       Queue& queue,
                       std::size_t block_size) {
      d_points = std::make_optional<PointsDevice>(queue, h_points.size());
      auto& dev_points = *d_points;

      setup(queue, h_points, dev_points, block_size);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    }
    template <typename KernelType>
    void make_clusters(PointsHost& h_points, const KernelType& kernel, std::size_t block_size) {
      auto device = alpaka::getDevByIdx(Platform{}, 0u);
      Queue queue(device);
      init_device(queue);

      d_points = std::make_optional<PointsDevice>(queue, h_points.size());
      auto& dev_points = *d_points;

      setup(queue, h_points, dev_points, block_size);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    }
    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& dev_points,
                       const KernelType& kernel,
                       Queue& queue,
                       std::size_t block_size) {
      setup(queue, h_points, dev_points, block_size);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    }
    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& dev_points,
                       const KernelType& kernel,
                       std::size_t block_size) {
      auto device = alpaka::getDevByIdx(Platform{}, 0u);
      Queue queue(device);
      init_device(queue);

      setup(queue, h_points, dev_points, block_size);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
    }

    void setWrappedCoordinates(const std::array<uint8_t, Ndim>& wrappedCoordinates) {
      m_wrappedCoordinates = wrappedCoordinates;
    }
    void setWrappedCoordinates(std::array<uint8_t, Ndim>&& wrappedCoordinates) {
      m_wrappedCoordinates = std::move(wrappedCoordinates);
    }
    template <typename... TArgs>
    void setWrappedCoordinates(TArgs... wrappedCoordinates) {
      m_wrappedCoordinates = {wrappedCoordinates...};
    }

    std::vector<std::vector<int>> getClusters(const PointsHost& h_points);

  private:
    float m_dc;
    float m_seed_dc;
    float m_rhoc;
    float m_dm;
    int m_pointsPerTile;  // average number of points found in a tile
    std::array<uint8_t, Ndim> m_wrappedCoordinates;

    // internal buffers
    std::optional<TilesDevice> d_tiles;
    std::optional<clue::device_buffer<Device, VecArray<int32_t, reserve>>> d_seeds;
    std::optional<FollowersDevice> d_followers;
    std::optional<PointsDevice> d_points;

    void init_device(Queue& queue);
    void init_device(Queue& queue, TilesDevice* tile_buffer);

    void setupTiles(Queue& queue, const PointsHost& h_points);
    void setupFollowers(Queue& queue, const PointsHost& h_points);
    void setupPoints(const PointsHost& h_points,
                     PointsDevice& dev_points,
                     Queue& queue,
                     std::size_t block_size);

    void setup(Queue& queue,
               const PointsHost& h_points,
               PointsDevice& dev_points,
               std::size_t block_size) {
      setupTiles(queue, h_points);
      setupFollowers(queue, h_points);
      setupPoints(h_points, dev_points, queue, block_size);
    }

    void calculate_tile_size(CoordinateExtremes* min_max,
                             float* tile_sizes,
                             const PointsHost& h_points,
                             int32_t nPerDim);

    template <typename KernelType>
    void make_clusters_impl(PointsHost& h_points,
                            PointsDevice& dev_points,
                            const KernelType& kernel,
                            Queue& queue,
                            std::size_t block_size);
  };

  template <uint8_t Ndim>
  void Clusterer<Ndim>::calculate_tile_size(CoordinateExtremes* min_max,
                                            float* tile_sizes,
                                            const PointsHost& h_points,
                                            int32_t nPerDim) {
    for (size_t dim{}; dim != Ndim; ++dim) {
      const float dimMax = *std::ranges::max_element(h_points.coords(dim));
      const float dimMin = *std::ranges::min_element(h_points.coords(dim));

      min_max->min(dim) = dimMin;
      min_max->max(dim) = dimMax;

      const float tileSize{(dimMax - dimMin) / nPerDim};
      tile_sizes[dim] = tileSize;
    }
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::init_device(Queue& queue) {
    d_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue);
    m_seeds = (*d_seeds).data();
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::init_device(Queue& queue, TilesDevice* tile_buffer) {
    d_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue);
    m_seeds = (*d_seeds).data();

    // load tiles from outside
    d_tiles = *tile_buffer;
    m_tiles = tile_buffer->view();
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupTiles(Queue& queue, const PointsHost& h_points) {
    // TODO: reconsider the way that we compute the number of tiles
    auto nTiles =
        static_cast<int32_t>(std::ceil(h_points.size() / static_cast<float>(m_pointsPerTile)));
    const auto nPerDim = static_cast<int32_t>(std::ceil(std::pow(nTiles, 1. / Ndim)));
    nTiles = static_cast<int32_t>(std::pow(nPerDim, Ndim));

    if (!d_tiles.has_value()) {
      d_tiles = std::make_optional<TilesDevice>(queue, h_points.size(), nTiles);
      m_tiles = d_tiles->view();
    }
    // check if tiles are large enough for current data
    if (!(alpaka::trait::GetExtents<clue::device_buffer<Device, int32_t[]>>{}(
              d_tiles->indexes())[0u] >= static_cast<std::size_t>(h_points.size())) or
        !(alpaka::trait::GetExtents<clue::device_buffer<Device, int32_t[]>>{}(
              d_tiles->offsets())[0u] >= static_cast<std::size_t>(nTiles))) {
      d_tiles->initialize(h_points.size(), nTiles, nPerDim, queue);
    } else {
      d_tiles->reset(h_points.size(), nTiles, nPerDim, queue);
    }

    auto min_max = clue::make_host_buffer<CoordinateExtremes>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);
    calculate_tile_size(min_max.data(), tile_sizes.data(), h_points, nPerDim);

    const auto device = alpaka::getDev(queue);
    alpaka::memcpy(queue, d_tiles->minMax(), min_max);
    alpaka::memcpy(queue, d_tiles->tileSize(), tile_sizes);
    alpaka::memcpy(
        queue, d_tiles->wrapped(), clue::make_host_view(m_wrappedCoordinates.data(), Ndim));
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupFollowers(Queue& queue, const PointsHost& h_points) {
    if (!d_followers.has_value()) {
      d_followers = std::make_optional<FollowersDevice>(h_points.size(), queue);
      m_followers = d_followers->view();
    }

    if (!(d_followers->extents() >= h_points.size())) {
      d_followers->initialize(h_points.size(), queue);
    } else {
      d_followers->reset(h_points.size(), queue);
    }
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupPoints(const PointsHost& h_points,
                                    PointsDevice& dev_points,
                                    Queue& queue,
                                    std::size_t block_size) {
    clue::copyToDevice(queue, dev_points, h_points);

    alpaka::memset(queue, *d_seeds, 0x00);
  }

  template <uint8_t Ndim>
  template <typename KernelType>
  void Clusterer<Ndim>::make_clusters_impl(PointsHost& h_points,
                                           PointsDevice& dev_points,
                                           const KernelType& kernel,
                                           Queue& queue,
                                           std::size_t block_size) {
    const auto nPoints = h_points.size();
    // fill the tiles
    d_tiles->template fill<Acc1D>(queue, dev_points, nPoints);

    const Idx grid_size = clue::divide_up_by(nPoints, block_size);
    auto working_div = clue::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::exec<Acc1D>(queue,
                        working_div,
                        KernelCalculateLocalDensity{},
                        m_tiles,
                        dev_points.view(),
                        kernel,
                        m_dc,
                        nPoints);
    alpaka::exec<Acc1D>(queue,
                        working_div,
                        KernelCalculateNearestHigher{},
                        m_tiles,
                        dev_points.view(),
                        m_dm,
                        nPoints);

    d_followers->template fill<Acc1D>(queue, dev_points);

    alpaka::exec<Acc1D>(queue,
                        working_div,
                        KernelFindClusters{},
                        m_seeds,
                        dev_points.view(),
                        m_seed_dc,
                        m_rhoc,
                        nPoints);

    // We change the working division when assigning the clusters
    const Idx grid_size_seeds = clue::divide_up_by(reserve, block_size);
    auto working_div_seeds = clue::make_workdiv<Acc1D>(grid_size_seeds, block_size);

    alpaka::exec<Acc1D>(
        queue, working_div_seeds, KernelAssignClusters{}, m_seeds, m_followers, dev_points.view());
    alpaka::wait(queue);

    clue::copyToHost(queue, h_points, dev_points);
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  std::vector<std::vector<int>> Clusterer<Ndim>::getClusters(const PointsHost& h_points) {
    return clue::compute_clusters_points(h_points.clusterIndexes());
  }

}  // namespace clue
