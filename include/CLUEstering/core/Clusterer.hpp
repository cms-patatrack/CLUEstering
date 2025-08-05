
#pragma once

#include "CLUEstering/core/CLUEAlpakaKernels.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/Tiles.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include "CLUEstering/utils/validation.hpp"

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
    using CoordinateExtremes = clue::CoordinateExtremes<Ndim>;
    using PointsHost = clue::PointsHost<Ndim>;
    using PointsDevice = clue::PointsDevice<Ndim, clue::Device>;
    using TilesDevice = clue::TilesAlpaka<Ndim, clue::Device>;
    using FollowersDevice = clue::Followers<clue::Device>;
    using Acc = internal::Acc;

    inline static constexpr auto reserve = internal::reserve;

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

      setup(queue, h_points, dev_points);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
      alpaka::wait(queue);
    }
    template <typename KernelType>
    void make_clusters(PointsHost& h_points, const KernelType& kernel, std::size_t block_size) {
      auto device = alpaka::getDevByIdx(Platform{}, 0u);
      Queue queue(device);
      init_device(queue);

      d_points = std::make_optional<PointsDevice>(queue, h_points.size());
      auto& dev_points = *d_points;

      setup(queue, h_points, dev_points);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
      alpaka::wait(queue);
    }
    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& dev_points,
                       const KernelType& kernel,
                       Queue& queue,
                       std::size_t block_size) {
      setup(queue, h_points, dev_points);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
      alpaka::wait(queue);
    }
    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& dev_points,
                       const KernelType& kernel,
                       std::size_t block_size) {
      auto device = alpaka::getDevByIdx(Platform{}, 0u);
      Queue queue(device);
      init_device(queue);

      setup(queue, h_points, dev_points);
      make_clusters_impl(h_points, dev_points, kernel, queue, block_size);
      alpaka::wait(queue);
    }
    template <typename KernelType>
    void make_clusters(PointsDevice& dev_points,
                       const KernelType& kernel,
                       Queue& queue,
                       std::size_t block_size) {
      setupTiles(queue, dev_points);
      setupFollowers(queue, dev_points.size());
      alpaka::memset(queue, *d_seeds, 0x00);
      make_clusters_impl(dev_points, kernel, queue, block_size);
      alpaka::wait(queue);
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
    void setupTiles(Queue& queue, const PointsDevice& d_points);

    void setupFollowers(Queue& queue, int32_t n_points);

    void setupPoints(const PointsHost& h_points, PointsDevice& dev_points, Queue& queue);

    void setup(Queue& queue, const PointsHost& h_points, PointsDevice& dev_points) {
      setupTiles(queue, h_points);
      setupFollowers(queue, h_points.size());
      setupPoints(h_points, dev_points, queue);
    }

    void calculate_tile_size(CoordinateExtremes* min_max,
                             float* tile_sizes,
                             const PointsHost& h_points,
                             int32_t nPerDim);
    void calculate_tile_size(Queue& queue,
                             CoordinateExtremes* min_max,
                             float* tile_sizes,
                             const PointsDevice& dev_points,
                             uint32_t nPerDim);

    template <typename KernelType>
    void make_clusters_impl(PointsHost& h_points,
                            PointsDevice& dev_points,
                            const KernelType& kernel,
                            Queue& queue,
                            std::size_t block_size);
    template <typename KernelType>
    void make_clusters_impl(PointsDevice& dev_points,
                            const KernelType& kernel,
                            Queue& queue,
                            std::size_t block_size);
  };

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
  void Clusterer<Ndim>::calculate_tile_size(Queue& queue,
                                            CoordinateExtremes* min_max,
                                            float* tile_sizes,
                                            const PointsDevice& dev_points,
                                            uint32_t nPerDim) {
    for (size_t dim{}; dim != Ndim; ++dim) {
      auto coords = dev_points.coords(dim);
      const auto* dimMax =
          clue::internal::algorithm::max_element(coords.data(), coords.data() + coords.size());
      const auto* dimMin =
          clue::internal::algorithm::min_element(coords.data(), coords.data() + coords.size());

      auto h_dimMin = make_host_buffer<float>(queue);
      auto h_dimMax = make_host_buffer<float>(queue);
      alpaka::memcpy(queue, h_dimMin, make_device_view(alpaka::getDev(queue), *dimMin));
      alpaka::memcpy(queue, h_dimMax, make_device_view(alpaka::getDev(queue), *dimMax));
      alpaka::wait(queue);

      min_max->min(dim) = *h_dimMin;
      min_max->max(dim) = *h_dimMax;

      const float tileSize{(*h_dimMax - *h_dimMin) / nPerDim};
      tile_sizes[dim] = tileSize;
    }
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

    alpaka::memcpy(queue, d_tiles->minMax(), min_max);
    alpaka::memcpy(queue, d_tiles->tileSize(), tile_sizes);
    alpaka::memcpy(
        queue, d_tiles->wrapped(), clue::make_host_view(m_wrappedCoordinates.data(), Ndim));
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupTiles(Queue& queue, const PointsDevice& d_points) {
    auto nTiles =
        static_cast<int32_t>(std::ceil(d_points.size() / static_cast<float>(m_pointsPerTile)));
    const auto nPerDim = static_cast<int32_t>(std::ceil(std::pow(nTiles, 1. / Ndim)));
    nTiles = static_cast<int32_t>(std::pow(nPerDim, Ndim));

    if (!d_tiles.has_value()) {
      d_tiles = std::make_optional<TilesDevice>(queue, d_points.size(), nTiles);
      m_tiles = d_tiles->view();
    }
    // check if tiles are large enough for current data
    if (!(alpaka::trait::GetExtents<clue::device_buffer<clue::Device, int32_t[]>>{}(
              d_tiles->indexes())[0u] >= d_points.size()) or
        !(alpaka::trait::GetExtents<clue::device_buffer<clue::Device, int32_t[]>>{}(
              d_tiles->offsets())[0u] >= static_cast<uint32_t>(nTiles))) {
      d_tiles->initialize(d_points.size(), nTiles, nPerDim, queue);
    } else {
      d_tiles->reset(d_points.size(), nTiles, nPerDim, queue);
    }

    auto min_max = clue::make_host_buffer<CoordinateExtremes>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);
    calculate_tile_size(queue, min_max.data(), tile_sizes.data(), d_points, nPerDim);

    alpaka::memcpy(queue, d_tiles->minMax(), min_max);
    alpaka::memcpy(queue, d_tiles->tileSize(), tile_sizes);
    alpaka::memcpy(
        queue, d_tiles->wrapped(), clue::make_host_view(m_wrappedCoordinates.data(), Ndim));
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupFollowers(Queue& queue, int32_t n_points) {
    if (!d_followers.has_value()) {
      d_followers = std::make_optional<FollowersDevice>(n_points, queue);
      m_followers = d_followers->view();
    }

    if (!(d_followers->extents() >= n_points)) {
      d_followers->initialize(n_points, queue);
    } else {
      d_followers->reset(n_points, queue);
    }
  }

  template <uint8_t Ndim>
  void Clusterer<Ndim>::setupPoints(const PointsHost& h_points,
                                    PointsDevice& dev_points,
                                    Queue& queue) {
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
    d_tiles->template fill<Acc>(queue, dev_points, nPoints);

    const Idx grid_size = clue::divide_up_by(nPoints, block_size);
    auto working_div = clue::make_workdiv<Acc>(grid_size, block_size);
    alpaka::exec<Acc>(queue,
                      working_div,
                      internal::KernelCalculateLocalDensity{},
                      m_tiles,
                      dev_points.view(),
                      kernel,
                      m_dc,
                      nPoints);
    alpaka::exec<Acc>(queue,
                      working_div,
                      internal::KernelCalculateNearestHigher{},
                      m_tiles,
                      dev_points.view(),
                      m_dm,
                      nPoints);

    d_followers->template fill<Acc>(queue, dev_points);

    alpaka::exec<Acc>(queue,
                      working_div,
                      internal::KernelFindClusters{},
                      m_seeds,
                      dev_points.view(),
                      m_seed_dc,
                      m_rhoc,
                      nPoints);

    // We change the working division when assigning the clusters
    const Idx grid_size_seeds = clue::divide_up_by(reserve, block_size);
    auto working_div_seeds = clue::make_workdiv<Acc>(grid_size_seeds, block_size);

    alpaka::exec<Acc>(queue,
                      working_div_seeds,
                      internal::KernelAssignClusters{},
                      m_seeds,
                      m_followers,
                      dev_points.view());
    alpaka::wait(queue);

    clue::copyToHost(queue, h_points, dev_points);
  }

  template <uint8_t Ndim>
  template <typename KernelType>
  void Clusterer<Ndim>::make_clusters_impl(PointsDevice& dev_points,
                                           const KernelType& kernel,
                                           Queue& queue,
                                           std::size_t block_size) {
    const auto nPoints = dev_points.size();
    d_tiles->template fill<Acc>(queue, dev_points, nPoints);

    const Idx grid_size = clue::divide_up_by(nPoints, block_size);
    auto working_div = clue::make_workdiv<Acc>(grid_size, block_size);
    alpaka::exec<Acc>(queue,
                      working_div,
                      internal::KernelCalculateLocalDensity{},
                      m_tiles,
                      dev_points.view(),
                      kernel,
                      m_dc,
                      nPoints);
    alpaka::exec<Acc>(queue,
                      working_div,
                      internal::KernelCalculateNearestHigher{},
                      m_tiles,
                      dev_points.view(),
                      m_dm,
                      nPoints);

    d_followers->template fill<Acc>(queue, dev_points);

    alpaka::exec<Acc>(queue,
                      working_div,
                      internal::KernelFindClusters{},
                      m_seeds,
                      dev_points.view(),
                      m_seed_dc,
                      m_rhoc,
                      nPoints);

    // We change the working division when assigning the clusters
    const Idx grid_size_seeds = clue::divide_up_by(reserve, block_size);
    auto working_div_seeds = clue::make_workdiv<Acc>(grid_size_seeds, block_size);

    alpaka::exec<Acc>(queue,
                      working_div_seeds,
                      internal::KernelAssignClusters{},
                      m_seeds,
                      m_followers,
                      dev_points.view());
  }

  template <uint8_t Ndim>
  std::vector<std::vector<int>> Clusterer<Ndim>::getClusters(const PointsHost& h_points) {
    return clue::compute_clusters_points(h_points.clusterIndexes());
  }

}  // namespace clue
