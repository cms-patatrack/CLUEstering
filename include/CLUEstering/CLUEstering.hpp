
#pragma once

#include <algorithm>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/Points.hpp"
#include "DataFormats/alpaka/PointsAlpaka.hpp"
#include "DataFormats/alpaka/TilesAlpaka.hpp"
#include "CLUE/CLUEAlpakaKernels.hpp"
#include "CLUE/ConvolutionalKernel.hpp"

using clue::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  template <uint8_t Ndim>
  class CLUEAlgoAlpaka {
  public:
    CLUEAlgoAlpaka() = delete;
    explicit CLUEAlgoAlpaka(float dc, float rhoc, float dm, int pPBin, Queue queue)
        : dc_{dc}, rhoc_{rhoc}, dm_{dm}, pointsPerTile_{pPBin} {
      init_device(queue);
    }

    TilesAlpakaView<Ndim>* m_tiles;
    VecArray<int32_t, reserve>* m_seeds;
    VecArray<int32_t, max_followers>* m_followers;

    template <typename KernelType>
    void make_clusters(PointsSoA<Ndim>& h_points,
                       const KernelType& kernel,
                       Queue queue,
                       std::size_t block_size);

    template <typename KernelType>
    void make_clusters(PointsSoA<Ndim>& h_points,
                       PointsAlpaka<Ndim>& dev_points,
                       const KernelType& kernel,
                       Queue queue,
                       std::size_t block_size);

    std::map<int, std::vector<int>> getClusters(const PointsSoA<Ndim>& h_points);

  private:
    float dc_;
    float rhoc_;
    float dm_;
    // average number of points found in a tile
    int pointsPerTile_;

    // Buffers
    std::unique_ptr<TilesAlpaka<Ndim>> d_tiles;
    std::optional<clue::device_buffer<Device, VecArray<int32_t, reserve>>> d_seeds;
    std::optional<clue::device_buffer<Device, clue::VecArray<int32_t, max_followers>[]>>
        d_followers;
    std::optional<PointsAlpaka<Ndim>> d_points;

    // Private methods
    void init_device(Queue queue);
    void setupTiles(Queue queue, const PointsSoA<Ndim>& h_points);
    void setupPoints(const PointsSoA<Ndim>& h_points,
                     PointsAlpaka<Ndim>& dev_points,
                     Queue queue,
                     std::size_t block_size);

    // Construction of the tiles
    void calculate_tile_size(CoordinateExtremes<Ndim>* min_max,
                             float* tile_sizes,
                             const PointsSoA<Ndim>& h_points,
                             uint32_t nPerDim);
  };

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::calculate_tile_size(CoordinateExtremes<Ndim>* min_max,
                                                 float* tile_sizes,
                                                 const PointsSoA<Ndim>& h_points,
                                                 uint32_t nPerDim) {
    auto n = h_points.nPoints();
    for (size_t dim{}; dim != Ndim; ++dim) {
      const auto offset = dim * n;
      const float dimMax =
          *std::max_element(h_points.coords() + offset, h_points.coords() + offset + n);
      const float dimMin =
          *std::min_element(h_points.coords() + offset, h_points.coords() + offset + n);

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
  void CLUEAlgoAlpaka<Ndim>::setupTiles(Queue queue, const PointsSoA<Ndim>& h_points) {
    // TODO: reconsider the way that we compute the number of tiles
    auto nTiles = static_cast<int32_t>(
        std::ceil(h_points.nPoints() / static_cast<float>(pointsPerTile_)));
    const auto nPerDim = static_cast<int32_t>(std::ceil(std::pow(nTiles, 1. / Ndim)));
    nTiles = static_cast<int32_t>(std::pow(nPerDim, Ndim));

    // TODO: check if nullptr and if not, reset without allocating
    d_tiles =
        std::make_unique<TilesAlpaka<Ndim>>(queue, h_points.nPoints(), nPerDim, nTiles);
    m_tiles = d_tiles->view();

    auto min_max = clue::make_host_buffer<CoordinateExtremes<Ndim>>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);
    calculate_tile_size(min_max.data(), tile_sizes.data(), h_points, nPerDim);

    // these are now private members
    const auto device = alpaka::getDev(queue);
    alpaka::memcpy(queue, d_tiles->minMax(), min_max);
    alpaka::memcpy(queue, d_tiles->tileSize(), tile_sizes);
    alpaka::memcpy(queue, d_tiles->wrapped(), clue::make_host_view(h_points.wrapped(), Ndim));
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::setupPoints(const PointsSoA<Ndim>& h_points,
                                         PointsAlpaka<Ndim>& dev_points,
                                         Queue queue,
                                         std::size_t block_size) {
    const auto copyExtent = (Ndim + 1) * h_points.nPoints();
    alpaka::memcpy(queue,
                   dev_points.input_buffer,
                   clue::make_host_view(h_points.coords(), copyExtent),
                   copyExtent);

    // TODO: when reworking the followers with the association map, this piece of
    // code will need to be moved
    alpaka::memset(queue, *d_seeds, 0x00);
    const Idx grid_size = clue::divide_up_by(h_points.nPoints(), block_size);
    const auto working_div = clue::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::exec<Acc1D>(
        queue, working_div, KernelResetFollowers{}, m_followers, h_points.nPoints());
  }

  template <uint8_t Ndim>
  template <typename KernelType>
  void CLUEAlgoAlpaka<Ndim>::make_clusters(PointsSoA<Ndim>& h_points,
                                           const KernelType& kernel,
                                           Queue queue,
                                           std::size_t block_size) {
    d_points = std::make_optional<PointsAlpaka<Ndim>>(queue, h_points.nPoints());
    auto& dev_points = *d_points;
    make_clusters(h_points, dev_points, kernel, queue, block_size);
  }

  template <uint8_t Ndim>
  template <typename KernelType>
  void CLUEAlgoAlpaka<Ndim>::make_clusters(PointsSoA<Ndim>& h_points,
                                           PointsAlpaka<Ndim>& dev_points,
                                           const KernelType& kernel,
                                           Queue queue,
                                           std::size_t block_size) {
    const auto device = alpaka::getDev(queue);
    setupTiles(queue, h_points);
    setupPoints(h_points, dev_points, queue, block_size);
    const auto nPoints = h_points.nPoints();

    // fill the tiles
    d_tiles->fill(queue, dev_points, nPoints);

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

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue);

#ifdef DEBUG
    alpaka::memcpy(queue,
                   clue::make_host_view(h_points.debugInfo().rho.data(), nPoints),
                   clue::make_device_view(device, dev_points.view()->rho, nPoints));
    alpaka::memcpy(queue,
                   clue::make_host_view(h_points.debugInfo().rho.data(), nPoints),
                   clue::make_device_view(device, dev_points.view()->delta, nPoints));
    alpaka::memcpy(
        queue,
        clue::make_host_view(h_points.debugInfo().nearestHigher.data(), nPoints),
        clue::make_device_view(device, dev_points.view()->nearest_higher, nPoints));
#endif

    alpaka::memcpy(queue,
                   clue::make_host_view(h_points.clusterIndexes(), 2 * nPoints),
                   clue::make_device_view(
                       device, dev_points.result_buffer.data() + nPoints, 2 * nPoints),
                   2 * nPoints);

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue);
  }

  template <uint8_t Ndim>
  std::map<int, std::vector<int>> CLUEAlgoAlpaka<Ndim>::getClusters(
      const PointsSoA<Ndim>& h_points) {
    // cluster all points with same clusterId
    std::map<int, std::vector<int>> clusters;
    for (size_t i = 0; i < h_points.nPoints(); i++) {
      clusters[h_points.clusterIndexes()[i]].push_back(i);
    }
    return clusters;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
