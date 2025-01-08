
#pragma once

#include <algorithm>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "../DataFormats/alpaka/TilesAlpaka.h"
#include "CLUEAlpakaKernels.h"
#include "ConvolutionalKernel.h"

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
    std::vector<std::vector<int>> make_clusters(Points<Ndim>& h_points,
                                                PointsAlpaka<Ndim>& d_points,
                                                const KernelType& kernel,
                                                Queue queue_,
                                                std::size_t block_size);

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

    // Private methods
    void init_device(Queue queue_);
    void setup(const Points<Ndim>& h_points,
               PointsAlpaka<Ndim>& d_points,
               Queue queue_,
               std::size_t block_size);

    // Construction of the tiles
    void calculate_tile_size(CoordinateExtremes<Ndim>& min_max,
                             float* tile_sizes,
                             const Points<Ndim>& h_points,
                             uint32_t nPerDim);
  };

  // Private methods
  template <uint8_t Ndim>
  void CLUEAlgoAlpaka<Ndim>::calculate_tile_size(CoordinateExtremes<Ndim>& min_max,
                                                 float* tile_sizes,
                                                 const Points<Ndim>& h_points,
                                                 uint32_t nPerDim) {
    for (size_t dim{}; dim != Ndim; ++dim) {
      const float dimMax{
          (*std::max_element(h_points.m_coords.begin(),
                             h_points.m_coords.end(),
                             [dim](const auto& vec1, const auto& vec2) -> bool {
                               return vec1[dim] < vec2[dim];
                             }))[dim]};
      const float dimMin{
          (*std::min_element(h_points.m_coords.begin(),
                             h_points.m_coords.end(),
                             [dim](const auto& vec1, const auto& vec2) -> bool {
                               return vec1[dim] < vec2[dim];
                             }))[dim]};

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
  void CLUEAlgoAlpaka<Ndim>::setup(const Points<Ndim>& h_points,
                                   PointsAlpaka<Ndim>& d_points,
                                   Queue queue_,
                                   std::size_t block_size) {
    // calculate the number of tiles and their size
    const auto nTiles{std::ceil(h_points.n / static_cast<float>(pointsPerTile_))};
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
    alpaka::memcpy(queue_,
                   d_points.coords,
                   clue::make_host_view(h_points.m_coords.data(), h_points.n));
    alpaka::memcpy(queue_,
                   d_points.weight,
                   clue::make_host_view(h_points.m_weight.data(), h_points.n));
    alpaka::memset(queue_, *d_seeds, 0x00);

    // Define the working division
    const Idx grid_size = clue::divide_up_by(h_points.n, block_size);
    const auto working_div = clue::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        working_div, KernelResetFollowers{}, m_followers, h_points.n));
  }

  // Public methods
  template <uint8_t Ndim>
  template <typename KernelType>
  std::vector<std::vector<int>> CLUEAlgoAlpaka<Ndim>::make_clusters(
      Points<Ndim>& h_points,
      PointsAlpaka<Ndim>& d_points,
      const KernelType& kernel,
      Queue queue_,
      std::size_t block_size) {
    setup(h_points, d_points, queue_, block_size);

    const Idx grid_size = clue::divide_up_by(h_points.n, block_size);
    auto working_div = clue::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(
            working_div, KernelFillTiles{}, d_points.view(), m_tiles, h_points.n));

    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateLocalDensity{},
                                                    m_tiles,
                                                    d_points.view(),
                                                    kernel,
                                                    /* m_domains.data(), */
                                                    dc_,
                                                    h_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateNearestHigher{},
                                                    m_tiles,
                                                    d_points.view(),
                                                    /* m_domains.data(), */
                                                    dm_,
                                                    dc_,
                                                    h_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelFindClusters<Ndim>{},
                                                    m_seeds,
                                                    m_followers,
                                                    d_points.view(),
                                                    dm_,
                                                    dc_,
                                                    rhoc_,
                                                    h_points.n));

    // We change the working division when assigning the clusters
    const Idx grid_size_seeds = clue::divide_up_by(reserve, block_size);
    auto working_div_seeds = clue::make_workdiv<Acc1D>(grid_size_seeds, block_size);

    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div_seeds,
                                                    KernelAssignClusters<Ndim>{},
                                                    m_seeds,
                                                    m_followers,
                                                    d_points.view()));

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue_);

#ifdef DEBUG
    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.m_rho.data(), h_points.n),
                   d_points.rho,
                   static_cast<uint32_t>(h_points.n));
    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.m_delta.data(), h_points.n),
                   d_points.delta,
                   static_cast<uint32_t>(h_points.n));
    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.m_nearestHigher.data(), h_points.n),
                   d_points.nearest_higher,
                   static_cast<uint32_t>(h_points.n));
#endif

    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.m_clusterIndex.data(), h_points.n),
                   d_points.cluster_index,
                   static_cast<uint32_t>(h_points.n));
    alpaka::memcpy(queue_,
                   clue::make_host_view(h_points.m_isSeed.data(), h_points.n),
                   d_points.is_seed,
                   static_cast<uint32_t>(h_points.n));

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue_);

    return {h_points.m_clusterIndex, h_points.m_isSeed};
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
