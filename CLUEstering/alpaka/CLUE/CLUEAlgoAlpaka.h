#ifndef CLUE_Algo_Alpaka_h
#define CLUE_Algo_Alpaka_h

#include <algorithm>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>
#include <utility>

#include "CLUEAlpakaKernels.h"
#include "ConvolutionalKernel.h"
#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "../DataFormats/alpaka/TilesAlpaka.h"
#include "../DataFormats/Math/DeltaPhi.h"

using cms::alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TAcc, uint8_t Ndim>
  class CLUEAlgoAlpaka {
  public:
    CLUEAlgoAlpaka() = delete;
    explicit CLUEAlgoAlpaka(float dc, float rhoc, float outlierDeltaFactor, int pPBin, Queue queue_)
        : dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor}, pointsPerTile_{pPBin} {
      init_device(queue_);
    }

    /* Points<Ndim> m_points_h; */
    TilesAlpaka<Ndim>* m_tiles;
    VecArray<uint32_t, max_seeds>* m_seeds;
    VecArray<uint32_t, max_followers>* m_followers;
    /* cms::alpakatools::device_buffer<Device, uint32_t[]> m_seeds; */
    /* cms::alpakatools::device_buffer<Device, VecArray<uint32_t, max_followers>> m_followers; */

    /* int calculateNTiles(int pointPerBin) { */
    /*   int ntiles{points_.n / pointPerBin}; */
    /*   try { */
    /* 	if (ntiles == 0) { */
    /* 	  throw 100; */
    /* 	} */
    /*   } catch (...) { */
    /* 	std::cout */
    /* 		<< "pointPerBin is set too high for you number of points. You must lower it in the clusterer constructor.\n"; */
    /*   } */
    /*   return ntiles; */
    /* } */

    /* std::array<float, Ndim> calculateTileSize(int NTiles, tiles<Ndim>& tiles_) { */
    /*   std::array<float, Ndim> tileSizes; */
    /*   int NperDim{static_cast<int>(std::pow(NTiles, 1.0 / Ndim))}; */

    /*   for (int i{}; i != Ndim; ++i) { */
    /* 	float tileSize; */
    /* 	float dimMax{*std::max_element(points_.coordinates_[i].begin(), points_.coordinates_[i].end())}; */
    /* 	float dimMin{*std::min_element(points_.coordinates_[i].begin(), points_.coordinates_[i].end())}; */
    /* 	tiles_.minMax[i] = {dimMin, dimMax}; */
    /* 	tileSize = (dimMax - dimMin) / NperDim; */

    /* 	tileSizes[i] = tileSize; */
    /*   } */
    /*   return tileSizes; */
    /* } */

    void make_clusters(const Points<Ndim>& h_points, PointsAlpaka<Ndim>& d_points, Queue queue_);
    /* { */
    /*   Tiles.nTiles = calculateNTiles(pointsPerTile_); */
    /*   Tiles.resizeTiles(); */
    /*   Tiles.tilesSize = calculateTileSize(Tiles.nTiles, Tiles); */

    /*   return {points_.clusterIndex, points_.isSeed}; */
    /* } */

  private:
    float dc_;
    float rhoc_;
    float outlierDeltaFactor_;
    // average number of points found in a tile
    int pointsPerTile_;

    // Buffers
    std::optional<cms::alpakatools::device_buffer<Device, TilesAlpaka<Ndim>>> d_tiles;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<uint32_t, max_seeds>>>
        d_seeds;
    std::optional<
        cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<uint32_t, max_followers>[]>>
        d_followers;

    // Private methods
    void init_device(Queue queue_);
    void setup(const Points<Ndim>& h_points, PointsAlpaka<Ndim>& d_points, Queue queue_);
  };

  template <typename TAcc, uint8_t Ndim>
  void CLUEAlgoAlpaka<TAcc, Ndim>::init_device(Queue queue_) {
    d_tiles = cms::alpakatools::make_device_buffer<TilesAlpaka<Ndim>>(queue_);
    d_seeds = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<uint32_t, max_seeds>>(queue_);
    d_followers =
        cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<uint32_t, max_followers>[]>(
            queue_, reserve);

    // Copy to the public pointers
    m_tiles = (*d_tiles).data();
    m_seeds = (*d_seeds).data();
    m_followers = (*d_followers).data();
  }

  template <typename TAcc, uint8_t Ndim>
  void CLUEAlgoAlpaka<TAcc, Ndim>::setup(const Points<Ndim>& h_points,
                                         PointsAlpaka<Ndim>& d_points,
                                         Queue queue_) {
    alpaka::memcpy(queue_,
                   d_points.coords,
                   cms::alpakatools::make_host_view(h_points.coords.data(), h_points.coords.size()));
    alpaka::memcpy(queue_,
                   d_points.weight,
                   cms::alpakatools::make_host_view(h_points.weight.data(), h_points.weight.size()));
    alpaka::memset(queue_, (*d_seeds), 0x00);

    // Define the working division
    const Idx block_size{1024};
    Idx grid_size = std::ceil(h_points.coords.size() / static_cast<float>(block_size));
    auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        working_div, KernelResetFollowers(), m_followers, h_points.coords.size()));
  }

  template <typename TAcc, uint8_t Ndim>
  void CLUEAlgoAlpaka<TAcc, Ndim>::make_clusters(const Points<Ndim>& h_points,
                                                 PointsAlpaka<Ndim>& d_points,
                                                 Queue queue_) {
    setup(h_points, d_points, queue_);

    const Idx block_size{1024};
    const Idx grid_size = std::ceil(h_points.coords.size() / static_cast<float>(block_size));
    auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        working_div, KernelFillTiles(), d_points.view(), m_tiles, h_points.coords.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateLocalDensity(),
                                                    m_tiles,
                                                    d_points.view(),
                                                    dc_,
                                                    h_points.coords.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateNearestHigher(),
                                                    m_tiles,
                                                    d_points.view(),
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    h_points.coords.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelFindClusters(),
                                                    m_seeds,
                                                    m_followers,
                                                    d_points.view(),
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    rhoc_,
                                                    h_points.coords.size()));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        working_div, KernelAssignClusters(), m_seeds, max_followers, d_points.view()));

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue_);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
