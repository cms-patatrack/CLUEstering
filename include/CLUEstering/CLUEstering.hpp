
#pragma once

#include "DataFormats/PointsHost.hpp"
#include "DataFormats/PointsDevice.hpp"
#include "DataFormats/alpaka/TilesAlpaka.hpp"
#include "CLUE/CLUEAlpakaKernels.hpp"
#include "CLUE/ConvolutionalKernel.hpp"
#include "utility/validation.hpp"

#include <algorithm>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  template <uint8_t Ndim>
  class CLUEAlgoAlpaka {
  public:
    using CoordinateExtremes = clue::CoordinateExtremes<Ndim>;
    using PointsHost = clue::PointsHost<Ndim>;
    using PointsDevice = clue::PointsDevice<Ndim, Device>;
    using TilesDevice = clue::TilesAlpaka<Ndim, Device>;

    explicit CLUEAlgoAlpaka(float dc, float rhoc, float dm, int pPBin, Queue queue)
        : dc_{dc}, rhoc_{rhoc}, dm_{dm}, pointsPerTile_{pPBin}, m_wrappedCoordinates{} {
      init_device(queue);
    }
    explicit CLUEAlgoAlpaka(
        float dc, float rhoc, float dm, int pPBin, Queue queue, TilesDevice* tile_buffer)
        : dc_{dc}, rhoc_{rhoc}, dm_{dm}, pointsPerTile_{pPBin}, m_wrappedCoordinates{} {
      init_device(queue, tile_buffer);
    }

    TilesAlpakaView<Ndim>* m_tiles;
    VecArray<int32_t, reserve>* m_seeds;
    VecArray<int32_t, max_followers>* m_followers;

    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       const KernelType& kernel,
                       Queue queue,
                       std::size_t block_size);
    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& d_points,
                       const KernelType& kernel,
                       Queue queue_,
                       std::size_t block_size);

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
    float dc_;
    float rhoc_;
    float dm_;
    // average number of points found in a tile
    int pointsPerTile_;
    std::array<uint8_t, Ndim> m_wrappedCoordinates;

    // internal buffers
    std::optional<TilesDevice> d_tiles;
    std::optional<clue::device_buffer<Device, VecArray<int32_t, reserve>>> d_seeds;
    std::optional<clue::device_buffer<Device, clue::VecArray<int32_t, max_followers>[]>>
        d_followers;
    std::optional<PointsDevice> d_points;

    void init_device(Queue queue_);
    void init_device(Queue queue_, TilesDevice* tile_buffer);

    void setupTiles(Queue queue, const PointsHost& h_points);
    void setupPoints(const PointsHost& h_points,
                     PointsDevice& dev_points,
                     Queue queue,
                     std::size_t block_size);

    void calculate_tile_size(CoordinateExtremes* min_max,
                             float* tile_sizes,
                             const PointsHost& h_points,
                             uint32_t nPerDim);
  };

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::calculate_tile_size(CoordinateExtremes* min_max,
                                                 float* tile_sizes,
                                                 const PointsHost& h_points,
                                                 uint32_t nPerDim) {
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
  void CLUEAlgoAlpaka<Ndim>::init_device(Queue queue) {
    d_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue);
    d_followers =
        clue::make_device_buffer<VecArray<int32_t, max_followers>[]>(queue, reserve);

    m_seeds = (*d_seeds).data();
    m_followers = (*d_followers).data();
  }

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::init_device(Queue queue_, TilesDevice* tile_buffer) {
    d_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue_);
    d_followers =
        clue::make_device_buffer<VecArray<int32_t, max_followers>[]>(queue_, reserve);

    m_seeds = (*d_seeds).data();
    m_followers = (*d_followers).data();

    // load tiles from outside
    d_tiles = *tile_buffer;
    m_tiles = tile_buffer->view();
  }

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::setupTiles(Queue queue, const PointsHost& h_points) {
    // TODO: reconsider the way that we compute the number of tiles
    auto nTiles = static_cast<int32_t>(
        std::ceil(h_points.size() / static_cast<float>(pointsPerTile_)));
    const auto nPerDim = static_cast<int32_t>(std::ceil(std::pow(nTiles, 1. / Ndim)));
    nTiles = static_cast<int32_t>(std::pow(nPerDim, Ndim));

    if (!d_tiles.has_value()) {
      d_tiles = std::make_optional<TilesDevice>(queue, h_points.size(), nTiles);
      m_tiles = d_tiles->view();
    }
    // check if tiles are large enough for current data
    if (!(alpaka::trait::GetExtents<clue::device_buffer<Device, uint32_t[]>>{}(
              d_tiles->indexes())[0u] >= h_points.size()) or
        !(alpaka::trait::GetExtents<clue::device_buffer<Device, uint32_t[]>>{}(
              d_tiles->offsets())[0u] >= static_cast<uint32_t>(nTiles))) {
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
    alpaka::memcpy(queue,
                   d_tiles->wrapped(),
                   clue::make_host_view(m_wrappedCoordinates.data(), Ndim));
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::setupPoints(const PointsHost& h_points,
                                         PointsDevice& dev_points,
                                         Queue queue,
                                         std::size_t block_size) {
    const auto copyExtent = (Ndim + 1) * h_points.size();
    alpaka::memcpy(queue,
                   clue::make_device_view(
                       alpaka::getDev(queue), dev_points.view()->coords, copyExtent),
                   clue::make_host_view(h_points.view()->coords, copyExtent),
                   copyExtent);

    // TODO: when reworking the followers with the association map, this piece of
    // code will need to be moved
    alpaka::memset(queue, *d_seeds, 0x00);
    const Idx grid_size = clue::divide_up_by(h_points.size(), block_size);
    const auto working_div = clue::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::exec<Acc1D>(
        queue, working_div, KernelResetFollowers{}, m_followers, h_points.size());
  }

  template <uint8_t Ndim>
  template <typename KernelType>
  void CLUEAlgoAlpaka<Ndim>::make_clusters(PointsHost& h_points,
                                           const KernelType& kernel,
                                           Queue queue,
                                           std::size_t block_size) {
    d_points = std::make_optional<PointsDevice>(queue, h_points.size());
    auto& dev_points = *d_points;
    make_clusters(h_points, dev_points, kernel, queue, block_size);
  }

  template <uint8_t Ndim>
  template <typename KernelType>
  void CLUEAlgoAlpaka<Ndim>::make_clusters(PointsHost& h_points,
                                           PointsDevice& dev_points,
                                           const KernelType& kernel,
                                           Queue queue,
                                           std::size_t block_size) {
    const auto device = alpaka::getDev(queue);
    setupTiles(queue, h_points);
    setupPoints(h_points, dev_points, queue, block_size);
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
                        dc_,
                        nPoints);
    alpaka::exec<Acc1D>(queue,
                        working_div,
                        KernelCalculateNearestHigher{},
                        m_tiles,
                        dev_points.view(),
                        dm_,
                        dc_,
                        nPoints);
    alpaka::exec<Acc1D>(queue,
                        working_div,
                        KernelFindClusters<Ndim>{},
                        m_seeds,
                        m_followers,
                        dev_points.view(),
                        dm_,
                        dc_,
                        rhoc_,
                        nPoints);

    // We change the working division when assigning the clusters
    const Idx grid_size_seeds = clue::divide_up_by(reserve, block_size);
    auto working_div_seeds = clue::make_workdiv<Acc1D>(grid_size_seeds, block_size);

    alpaka::exec<Acc1D>(queue,
                        working_div_seeds,
                        KernelAssignClusters<Ndim>{},
                        m_seeds,
                        m_followers,
                        dev_points.view());
    alpaka::wait(queue);

    alpaka::memcpy(
        queue,
        clue::make_host_view(h_points.view()->cluster_index, 2 * nPoints),
        clue::make_device_view(device, dev_points.view()->cluster_index, 2 * nPoints),
        2 * nPoints);
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  std::vector<std::vector<int>> CLUEAlgoAlpaka<Ndim>::getClusters(
      const PointsHost& h_points) {
    return clue::compute_clusters_points(h_points.clusterIndexes());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
