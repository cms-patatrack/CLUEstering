/// @file Clusterer.hpp
/// @brief Implements the Clusterer class, which is the interface for running the clustering algorithm.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/core/detail/SetupFollowers.hpp"
#include "CLUEstering/core/detail/SetupTiles.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/DeviceVector.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

namespace clue {

  /// @brief The Clusterer class is the interface for running the clustering algorithm.
  /// It provides methods to set up the clustering parameters, initializes the internal buffers
  /// and runs the clustering algorithm on host or device points.
  ///
  /// @tparam Ndim The number of dimensions of the points to cluster
  template <std::size_t Ndim, std::floating_point DataType = float>
  class Clusterer {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<DataType>>;

  private:
    value_type m_dc;
    value_type m_seed_dc;
    value_type m_rhoc;
    value_type m_dm;
    int m_pointsPerTile;  // average number of points found in a tile
    std::array<uint8_t, Ndim> m_wrappedCoordinates;

    std::optional<internal::Tiles<Ndim, value_type, clue::Device>> m_tiles;
    std::optional<internal::SeedArray<>> m_seeds;
    std::optional<Followers<clue::Device>> m_followers;
    std::optional<internal::DeviceVector<>> m_event_associations;

    template <std::floating_point InputType>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void setup(Queue& queue,
               const clue::PointsHost<Ndim, InputType>& h_points,
               clue::PointsDevice<Ndim, value_type>& dev_points) {
      detail::setup_tiles(queue, h_points, m_tiles, m_pointsPerTile, m_wrappedCoordinates);
      detail::setup_followers(queue, m_followers, h_points.size());
      clue::copyToDevice(queue, dev_points, h_points);
    }

    template <std::floating_point InputType>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void setup_batch(Queue& queue,
                     const clue::PointsHost<Ndim, InputType>& h_points,
                     clue::PointsDevice<Ndim, value_type>& dev_points,
                     std::size_t batch_size) {
      detail::setup_tiles(
          queue, h_points, m_tiles, m_pointsPerTile, m_wrappedCoordinates, batch_size);
      detail::setup_followers(queue, m_followers, h_points.size());
      clue::copyToDevice(queue, dev_points, h_points);
    }

    template <std::floating_point InputType>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void setup_batch(Queue& queue,
                     clue::PointsDevice<Ndim, InputType>& dev_points,
                     std::size_t batch_size) {
      detail::setup_tiles(
          queue, dev_points, m_tiles, m_pointsPerTile, m_wrappedCoordinates, batch_size);
      detail::setup_followers(queue, m_followers, dev_points.size());
    }

    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters_impl(clue::PointsHost<Ndim, InputType>& h_points,
                            clue::PointsDevice<Ndim, value_type>& dev_points,
                            const DistanceMetric& metric,
                            const Kernel& kernel,
                            Queue& queue,
                            std::size_t block_size);
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters_impl(clue::PointsDevice<Ndim, InputType>& dev_points,
                            const DistanceMetric& metric,
                            const Kernel& kernel,
                            Queue& queue,
                            std::size_t block_size);
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters_batched(clue::PointsDevice<Ndim, InputType>& dev_points,
                               std::span<const uint32_t> batch_item_sizes,
                               const DistanceMetric& metric,
                               const Kernel& kernel,
                               Queue& queue,
                               std::size_t block_size);

  public:
    /// @brief Constuct a Clusterer object
    ///
    /// @param dc Distance threshold for clustering.
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters. This parameter is optional and by default dc is used.
    /// @param seed_dc Distance threshold for seed points. This parameter is optional and by default dc is used.
    /// @param pPBin Number of points per bin, used to determine the tile size
    Clusterer(value_type dc,
              value_type rhoc,
              std::optional<value_type> dm = std::nullopt,
              std::optional<value_type> seed_dc = std::nullopt,
              int pPBin = 128);
    /// @brief Constuct a Clusterer object
    ///
    /// @param queue The queue to use for the device operations
    /// @param dc Distance threshold for clustering.
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters. This parameter is optional and by default dc is used.
    /// @param seed_dc Distance threshold for seed points. This parameter is optional and by default dc is used.
    /// @param pPBin Number of points per bin, used to determine the tile size
    Clusterer(Queue& queue,
              value_type dc,
              value_type rhoc,
              std::optional<value_type> dm = std::nullopt,
              std::optional<value_type> seed_dc = std::nullopt,
              int pPBin = 128);

    /// @brief Set the parameters for the clustering algorithm
    ///
    /// @param dc Distance threshold for clustering
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters. This parameter is optional and by default dc is used.
    /// @param seed_dc Distance threshold for seed points. This parameter is optional and by default dc is used.
    /// @param pPBin Number of points per bin, used to determine the tile size
    void setParameters(value_type dc,
                       value_type rhoc,
                       std::optional<value_type> dm = std::nullopt,
                       std::optional<value_type> seed_dc = std::nullopt,
                       int pPBin = 128);

    /// @brief Construct the clusters from host points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param h_points Host points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters(Queue& queue,
                       clue::PointsHost<Ndim, InputType>& h_points,
                       const DistanceMetric& metric = clue::EuclideanMetric<Ndim, value_type>{},
                       const Kernel& kernel = FlatKernel<>{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from host points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param h_points Host points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    /// @note This method creates a temporary queue for the operations on the device
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters(clue::PointsHost<Ndim, InputType>& h_points,
                       const DistanceMetric& metric = clue::EuclideanMetric<Ndim, value_type>{},
                       const Kernel& kernel = FlatKernel<>{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from host and device points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param h_points Host points to cluster
    /// @param dev_points Device points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters(Queue& queue,
                       clue::PointsHost<Ndim, InputType>& h_points,
                       clue::PointsDevice<Ndim, value_type>& dev_points,
                       const DistanceMetric& metric = clue::EuclideanMetric<Ndim, value_type>{},
                       const Kernel& kernel = FlatKernel<>{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from device points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param dev_points Device points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters(Queue& queue,
                       clue::PointsDevice<Ndim, InputType>& dev_points,
                       const DistanceMetric& metric = clue::EuclideanMetric<Ndim, value_type>{},
                       const Kernel& kernel = FlatKernel<>{.5f},
                       std::size_t block_size = 256);

    /// @brief Construct the clusters from batched host and device points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param h_points Host points to cluster
    /// @param dev_points Device points to cluster
    /// @param batch_item_sizes Sizes of each batch item
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    /// @note The total size of h_points and dev_points must be equal to the sum of batch_item_sizes
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters(Queue& queue,
                       clue::PointsHost<Ndim, InputType>& h_points,
                       clue::PointsDevice<Ndim, value_type>& dev_points,
                       std::span<const uint32_t> batch_item_sizes,
                       const DistanceMetric& metric = clue::EuclideanMetric<Ndim, value_type>{},
                       const Kernel& kernel = FlatKernel<>{.5f},
                       std::size_t block_size = 256);

    /// @brief Construct the clusters from batched device points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param dev_points Device points to cluster
    /// @param batch_item_sizes Sizes of each batch item
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    /// @note The total size of h_points and dev_points must be equal to the sum of batch_item_sizes
    template <
        std::floating_point InputType,
        concepts::convolutional_kernel Kernel = FlatKernel<>,
        concepts::distance_metric<Ndim> DistanceMetric = clue::EuclideanMetric<Ndim, value_type>>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    void make_clusters(Queue& queue,
                       clue::PointsDevice<Ndim, InputType>& dev_points,
                       std::span<const uint32_t> batch_item_sizes,
                       const DistanceMetric& metric = clue::EuclideanMetric<Ndim, value_type>{},
                       const Kernel& kernel = FlatKernel<>{.5f},
                       std::size_t block_size = 256);

    /// @brief Specify which coordinates are periodic
    ///
    /// @param wrappedCoordinates Array of wrapped coordinates, where 1 means periodic and 0 means non-periodic
    template <std::ranges::contiguous_range TRange>
      requires std::integral<std::ranges::range_value_t<TRange>>
    void setWrappedCoordinates(const TRange& wrapped_coordinates);
    /// @brief Specify which coordinates are periodic
    ///
    /// @tparam TArgs Types of the wrapped coordinates, should be convertible to uint8_t
    /// @param wrappedCoordinates Wrapped coordinates, where 1 means periodic and 0 means non-periodic
    template <std::integral... TArgs>
    void setWrappedCoordinates(TArgs... wrapped_coordinates);

    /// @brief Get the clusters from the host points
    ///
    /// @param h_points Host points
    /// @return An associator mapping clusters and points
    template <std::floating_point InputType>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    host_associator getClusters(const clue::PointsHost<Ndim, InputType>& h_points);
    /// @brief Get the clusters from the device points
    /// This function returns an associator object mapping the clusters to the points they contain.
    ///
    /// @param d_points Device points
    /// @return An associator mapping clusters and points
    template <std::floating_point InputType>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    AssociationMap<Device> getClusters(Queue& queue,
                                       const clue::PointsDevice<Ndim, InputType>& d_points);

    /// @brief Get the sample-to-cluster associations for batched clustering
    ///
    /// @param queue The queue to use for the device operations
    /// @return A device buffer containing the event associations
    template <std::floating_point InputType>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    host_associator getSampleAssociations(Queue& queue,
                                          clue::PointsHost<Ndim, InputType>& h_points);
    /// @brief Get the sample-to-cluster associations for batched clustering
    ///
    /// @param queue The queue to use for the device operations
    /// @return A device buffer containing the event associations
    template <std::floating_point InputType>
      // requires std::same_as<std::remove_cv_t<InputType>, value_type>
    AssociationMap<Device> getSampleAssociations(Queue& queue,
                                                 clue::PointsDevice<Ndim, InputType>& d_points);
  };

}  // namespace clue

#include "CLUEstering/core/detail/Clusterer.hpp"
