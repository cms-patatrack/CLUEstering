
#pragma once

#include <algorithm>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
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
    explicit CLUEAlgoAlpaka(float dc, float rhoc, float dm, int pPBin, Queue queue_)
        : dc_{dc}, rhoc_{rhoc}, dm_{dm}, pointsPerTile_{pPBin} {
      init_device(queue_);
    }

    TilesAlpaka<Ndim>* m_tiles;
    VecArray<int32_t, reserve>* m_seeds;
    VecArray<int32_t, max_followers>* m_followers;

    template <typename KernelType>
    void make_clusters(PointsSoA<Ndim>& h_points,
                       const KernelType& kernel,
                       Queue queue_,
                       std::size_t block_size);

    std::map<int, std::vector<int>> getClusters(const PointsSoA<Ndim>& h_points);

  private:
    float dc_;
    float rhoc_;
    float dm_;
    // average number of points found in a tile
    int pointsPerTile_;

    /* domain_t<Ndim> m_domains; */

    // Buffers
    std::optional<clue::device_buffer<Device, TilesAlpaka<Ndim>>> d_tiles;
    std::optional<clue::device_buffer<Device, VecArray<int32_t, reserve>>> d_seeds;
    std::optional<clue::device_buffer<Device, clue::VecArray<int32_t, max_followers>[]>>
        d_followers;
    std::optional<PointsAlpaka<Ndim>> d_points;

    // Private methods
    void init_device(Queue queue_);
    void setup(const PointsSoA<Ndim>& h_points,
               Queue queue_,
               std::size_t block_size);

    // Construction of the tiles
    void calculate_tile_size(CoordinateExtremes<Ndim>& min_max,
                             float* tile_sizes,
                             const PointsSoA<Ndim>& h_points,
                             uint32_t nPerDim);
  };

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::calculate_tile_size(CoordinateExtremes<Ndim>& min_max,
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

      min_max.min(dim) = dimMin;
      min_max.max(dim) = dimMax;

      const float tileSize{(dimMax - dimMin) / nPerDim};
      tile_sizes[dim] = tileSize;
    }
  }

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::init_device(Queue queue_) {
    d_tiles = clue::make_device_buffer<TilesAlpaka<Ndim>>(queue_);
    d_seeds = clue::make_device_buffer<VecArray<int32_t, reserve>>(queue_);
    d_followers =
        clue::make_device_buffer<VecArray<int32_t, max_followers>[]>(queue_, reserve);

    // Copy to the public pointers
    m_tiles = (*d_tiles).data();
    m_seeds = (*d_seeds).data();
    m_followers = (*d_followers).data();
  }

  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::setup(const PointsSoA<Ndim>& h_points,
                                   Queue queue_,
                                   std::size_t block_size) {

    d_points = PointsAlpaka<Ndim>(queue_, h_points.nPoints());

    // calculate the number of tiles and their size
    const auto nTiles{std::ceil(h_points.nPoints() / static_cast<float>(pointsPerTile_))};
    const auto nPerDim{std::ceil(std::pow(nTiles, 1. / Ndim))};

    CoordinateExtremes<Ndim> min_max;
    float tile_size[Ndim];
    calculate_tile_size(min_max, tile_size, h_points, nPerDim);

    const auto device = alpaka::getDev(queue_);
    alpaka::memcpy(queue_,
                   clue::make_device_view(device, (*d_tiles)->minMax(), 2 * Ndim),
                   clue::make_host_view(min_max.data(), 2 * Ndim));
    alpaka::memcpy(queue_,
                   clue::make_device_view(device, (*d_tiles)->tileSize(), Ndim),
                   clue::make_host_view(tile_size, Ndim));
    alpaka::wait(queue_);

    const Idx tiles_grid_size = clue::divide_up_by(nTiles, block_size);
    const auto tiles_working_div = clue::make_workdiv<Acc1D>(tiles_grid_size, block_size);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        tiles_working_div, KernelResetTiles{}, m_tiles, nTiles, nPerDim));

    const auto copyExtent = (Ndim + 1) * h_points.nPoints();
    alpaka::memcpy(queue_,
                   d_points->input_buffer,
                   clue::make_host_view(h_points.coords(), copyExtent),
                   copyExtent);
    alpaka::memset(queue_, *d_seeds, 0x00);

    // Define the working division
    const Idx grid_size = clue::divide_up_by(h_points.nPoints(), block_size);
    const auto working_div = clue::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(
            working_div, KernelResetFollowers{}, m_followers, h_points.nPoints()));
  }

  // Public methods
  template <uint8_t Ndim>
  template <typename KernelType>
  void CLUEAlgoAlpaka<Ndim>::make_clusters(PointsSoA<Ndim>& h_points,
                                           const KernelType& kernel,
                                           Queue queue_,
                                           std::size_t block_size) {
    setup(h_points, queue_, block_size);

    const auto nPoints = h_points.nPoints();

    const Idx grid_size = clue::divide_up_by(nPoints, block_size);
    auto working_div = clue::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(
            working_div, KernelFillTiles{}, d_points->view(), m_tiles, nPoints));

    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateLocalDensity{},
                                                    m_tiles,
                                                    d_points->view(),
                                                    kernel,
                                                    /* m_domains.data(), */
                                                    dc_,
                                                    nPoints));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateNearestHigher{},
                                                    m_tiles,
                                                    d_points->view(),
                                                    /* m_domains.data(), */
                                                    dm_,
                                                    dc_,
                                                    nPoints));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelFindClusters<Ndim>{},
                                                    m_seeds,
                                                    m_followers,
                                                    d_points->view(),
                                                    dm_,
                                                    dc_,
                                                    rhoc_,
                                                    nPoints));

    // We change the working division when assigning the clusters
    const Idx grid_size_seeds = clue::divide_up_by(reserve, block_size);
    auto working_div_seeds = clue::make_workdiv<Acc1D>(grid_size_seeds, block_size);

    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div_seeds,
                                                    KernelAssignClusters<Ndim>{},
                                                    m_seeds,
                                                    m_followers,
                                                    d_points->view()));

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue_);

    const auto device = alpaka::getDev(queue_);
#ifdef DEBUG
    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.debugInfo().rho.data(), nPoints),
                   clue::make_device_view(device, d_points->view()->rho, nPoints));
    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.debugInfo().rho.data(), nPoints),
                   clue::make_device_view(device, d_points->view()->delta, nPoints));
    alpaka::memcpy(
        queue_,
        clue::make_host_view(h_points.debugInfo().nearestHigher.data(), nPoints),
        clue::make_device_view(device, d_points->view()->nearest_higher, nPoints));
#endif

    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.clusterIndexes(), 2 * nPoints),
                   clue::make_device_view(
                       device, d_points->result_buffer.data() + nPoints, 2 * nPoints),
                   2 * nPoints);

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue_);
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
