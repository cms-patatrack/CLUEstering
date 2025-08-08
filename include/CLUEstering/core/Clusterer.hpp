
#pragma once

#include "CLUEstering/core/detail/CLUEAlpakaKernels.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/Tiles.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"

#include <cstdint>
#include <vector>

namespace clue {

  template <uint8_t Ndim>
  class Clusterer {
  private:
    using CoordinateExtremes = clue::CoordinateExtremes<Ndim>;
    using PointsHost = clue::PointsHost<Ndim>;
    using PointsDevice = clue::PointsDevice<Ndim, clue::Device>;
    using TilesDevice = clue::TilesAlpaka<Ndim, clue::Device>;
    using FollowersDevice = clue::Followers<clue::Device>;
    using Acc = internal::Acc;
    inline static constexpr auto reserve = detail::reserve;

    float m_dc;
    float m_seed_dc;
    float m_rhoc;
    float m_dm;
    int m_pointsPerTile;  // average number of points found in a tile
    std::array<uint8_t, Ndim> m_wrappedCoordinates;

    std::optional<TilesDevice> d_tiles;
    std::optional<clue::device_buffer<Device, VecArray<int32_t, reserve>>> d_seeds;
    std::optional<FollowersDevice> d_followers;
    std::optional<PointsDevice> d_points;

    TilesAlpakaView<Ndim>* m_tiles;
    VecArray<int32_t, reserve>* m_seeds;
    FollowersView* m_followers;

    void init_device(Queue& queue);
    void init_device(Queue& queue, TilesDevice* tile_buffer);

    void setupTiles(Queue& queue, const PointsHost& h_points);
    void setupTiles(Queue& queue, const PointsDevice& d_points);

    void setupFollowers(Queue& queue, int32_t n_points);

  public:
    Clusterer(float dc, float rhoc, float dm, float seed_dc = -1.f, int pPBin = 128);
    Clusterer(Queue& queue, float dc, float rhoc, float dm, float seed_dc = -1.f, int pPBin = 128);
    Clusterer(Queue& queue,
              TilesDevice* tile_buffer,
              float dc,
              float rhoc,
              float dm,
              float seed_dc = -1.f,
              int pPBin = 128);

    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       const KernelType& kernel,
                       Queue& queue,
                       std::size_t block_size);
    template <typename KernelType>
    void make_clusters(PointsHost& h_points, const KernelType& kernel, std::size_t block_size);
    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& dev_points,
                       const KernelType& kernel,
                       Queue& queue,
                       std::size_t block_size);
    template <typename KernelType>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& dev_points,
                       const KernelType& kernel,
                       std::size_t block_size);
    template <typename KernelType>
    void make_clusters(PointsDevice& dev_points,
                       const KernelType& kernel,
                       Queue& queue,
                       std::size_t block_size);

    void setWrappedCoordinates(const std::array<uint8_t, Ndim>& wrappedCoordinates);
    void setWrappedCoordinates(std::array<uint8_t, Ndim>&& wrappedCoordinates);
    template <typename... TArgs>
    void setWrappedCoordinates(TArgs... wrappedCoordinates);

    std::vector<std::vector<int>> getClusters(const PointsHost& h_points);

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

}  // namespace clue

#include "CLUEstering/core/detail/Clusterer.hpp"
