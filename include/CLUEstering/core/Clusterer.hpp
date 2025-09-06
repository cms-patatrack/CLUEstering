/// @file Clusterer.hpp
/// @brief Implements the Clusterer class, which is the interface for running the clustering algorithm.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

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

  /// @brief The Clusterer class is the interface for running the clustering algorithm.
  /// It provides methods to set up the clustering parameters, initializes the internal buffers
  /// and runs the clustering algorithm on host or device points.
  ///
  /// @tparam Ndim The number of dimensions of the points to cluster
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

    std::optional<TilesDevice> m_tiles;
    std::optional<clue::device_buffer<Device, VecArray<int32_t, reserve>>> m_seeds;
    std::optional<FollowersDevice> m_followers;
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

    template <concepts::convolutional_kernel Kernel>
    void make_clusters_impl(PointsHost& h_points,
                            PointsDevice& dev_points,
                            const Kernel& kernel,
                            Queue& queue,
                            std::size_t block_size);
    template <concepts::convolutional_kernel Kernel>
    void make_clusters_impl(PointsDevice& dev_points,
                            const Kernel& kernel,
                            Queue& queue,
                            std::size_t block_size);

  public:
    /// @brief Constuct a Clusterer object
    ///
    /// @param dc Distance threshold for clustering
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters
    /// @param seed_dc Distance threshold for seed points, if -1.f, dc is used
    /// @param pPBin Number of points per bin, used to determine the tile size
    Clusterer(float dc, float rhoc, float dm, float seed_dc = -1.f, int pPBin = 128);
    /// @brief Constuct a Clusterer object
    ///
    /// @param queue The queue to use for the device operations
    /// @param dc Distance threshold for clustering
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters
    /// @param seed_dc Distance threshold for seed points, if the default value -1.f, dc is used
    /// @param pPBin Number of points per bin, used to determine the tile size
    Clusterer(Queue& queue, float dc, float rhoc, float dm, float seed_dc = -1.f, int pPBin = 128);
    /// @brief Constuct a Clusterer object
    ///
    /// @param queue The queue to use for the device operations
    /// @param tile_buffer Buffer to pre-allocated tiles
    /// @param dc Distance threshold for clustering
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters
    /// @param seed_dc Distance threshold for seed points, if the default value -1.f, dc is used
    /// @param pPBin Number of points per bin, used to determine the tile size
    Clusterer(Queue& queue,
              TilesDevice* tile_buffer,
              float dc,
              float rhoc,
              float dm,
              float seed_dc = -1.f,
              int pPBin = 128);

    /// @brief Set the parameters for the clustering algorithm
    ///
    /// @param dc Distance threshold for clustering
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters
    /// @param seed_dc Distance threshold for seed points, if the defualt value -1.f, dc is used
    /// @param pPBin Number of points per bin, used to determine the tile size
    void setParameters(float dc, float rhoc, float dm, float seed_dc = -1.f, int pPBin = 128);

    /// @brief Construct the clusters from host points
    ///
    /// @param queue The queue to use for the device operations
    /// @param h_points Host points to cluster
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <concepts::convolutional_kernel Kernel = FlatKernel>
    void make_clusters(Queue& queue,
                       PointsHost& h_points,
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from host points
    ///
    /// @param h_points Host points to cluster
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    /// @note This method creates a temporary queue for the operations on the device
    template <concepts::convolutional_kernel Kernel = FlatKernel>
    void make_clusters(PointsHost& h_points,
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from host and device points
    ///
    /// @param queue The queue to use for the device operations
    /// @param h_points Host points to cluster
    /// @param dev_points Device points to cluster
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <concepts::convolutional_kernel Kernel = FlatKernel>
    void make_clusters(Queue& queue,
                       PointsHost& h_points,
                       PointsDevice& dev_points,
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from host and device points
    ///
    /// @param h_points Host points to cluster
    /// @param dev_points Device points to cluster
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    /// @note This method creates a temporary queue for the operations on the device
    template <concepts::convolutional_kernel Kernel = FlatKernel>
    void make_clusters(PointsHost& h_points,
                       PointsDevice& dev_points,
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from device points
    ///
    /// @param queue The queue to use for the device operations
    /// @param dev_points Device points to cluster
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <concepts::convolutional_kernel Kernel = FlatKernel>
    void make_clusters(Queue& queue,
                       PointsDevice& dev_points,
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);

    /// @brief Specify which coordinates are periodic
    ///
    /// @param wrappedCoordinates Array of wrapped coordinates, where 1 means periodic and 0 means non-periodic
    void setWrappedCoordinates(const std::array<uint8_t, Ndim>& wrappedCoordinates);
    /// @brief Specify which coordinates are periodic
    ///
    /// @param wrappedCoordinates Array of wrapped coordinates, where 1 means periodic and 0 means non-periodic
    void setWrappedCoordinates(std::array<uint8_t, Ndim>&& wrappedCoordinates);
    /// @brief Specify which coordinates are periodic
    ///
    /// @tparam TArgs Types of the wrapped coordinates, should be convertible to uint8_t
    /// @param wrappedCoordinates Wrapped coordinates, where 1 means periodic and 0 means non-periodic
    template <typename... TArgs>
    void setWrappedCoordinates(TArgs... wrappedCoordinates);

    /// @brief Get the clusters from the host points
    ///
    /// @param h_points Host points to cluster
    /// @return A vector of clusters, where each cluster is a vector of point indices
    std::vector<std::vector<int>> getClusters(const PointsHost& h_points);
  };

}  // namespace clue

#include "CLUEstering/core/detail/Clusterer.hpp"
