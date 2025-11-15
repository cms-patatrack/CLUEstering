/// @file PointsHost.hpp
/// @brief Provides the PointsHost class for managing points in host memory
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/ClusterProperties.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <std::size_t NDim, concepts::queue TQueue>
  clue::PointsHost<NDim> read_output(TQueue& queue, const std::string& file_path);

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToHost(TQueue& queue,
                  PointsHost<Ndim>& h_points,
                  const PointsDevice<Ndim, TDev>& d_points);

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  auto copyToHost(TQueue& queue, const PointsDevice<Ndim, TDev>& d_points);

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToDevice(TQueue& queue,
                    PointsDevice<Ndim, TDev>& d_points,
                    const PointsHost<Ndim>& h_points);

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  auto copyToDevice(TQueue& queue, const PointsHost<Ndim>& h_points);

  /// @brief The PointsHost class is a data structure that manages points in host memory.
  /// It provides methods to allocate, access, and manipulate points in host memory.
  ///
  /// @tparam Ndim The number of dimensions of the points to manage
  template <std::size_t Ndim>
  class PointsHost : public internal::points_interface<PointsHost<Ndim>> {
  private:
    std::optional<host_buffer<std::byte[]>> m_buffer;
    PointsView<Ndim> m_view;
    std::optional<ClusterProperties> m_clusterProperties;
    std::optional<std::size_t> m_nclusters;
    int32_t m_size;
    bool m_clustered = false;

  public:
    class Point {
      std::array<float, Ndim> m_coordinates;
      float m_weight;
      int m_clusterIndex;

    public:
      Point(const std::array<float, Ndim>& coordinates, float weight, int cluster_index);
      float operator[](size_t dim) const;

      float weight() const;
      float cluster_index() const;
    };

    /// @brief Constructs a container for the points allocated on the host
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points to allocate
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points);

    /// @brief Constructs a container for the points allocated on the host using a pre-allocated buffers
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param buffer The pre-allocated buffer to use for the points data
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points, std::span<std::byte> buffer);

    /// @brief Constructs a container for the points allocated on the host using interleaved data
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param input_buffer The pre-allocated buffer containing interleaved coordinates and weights
    /// @param output_buffer The pre-allocated buffer to store the cluster indexes
    /// @note The input buffer must contain the coordinates and weights in an SoA format
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points, std::span<float> input, std::span<int> output);

    /// @brief Constructs a container for the points allocated on the host using separate coordinate and weight buffers
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param coordinates The pre-allocated buffer containing the coordinates
    /// @param weights The pre-allocated buffer containing the weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The coordinates buffer must have a size of n_points * Ndim
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue,
               int32_t n_points,
               std::span<float> coordinates,
               std::span<float> weights,
               std::span<int> output);

    /// @brief Constructs a container for the points allocated on the host using multiple pre-allocated buffers
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param buffers The pre-allocated buffers to use for the points data
    template <concepts::queue TQueue, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
    PointsHost(TQueue& queue, int32_t n_points, TBuffers&&... buffers);

    /// @brief Constructs a container for the points allocated on the host using interleaved data
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param input_buffer The pre-allocated buffer containing interleaved coordinates and weights
    /// @param output_buffer The pre-allocated buffer to store the cluster indexes
    /// @note The input buffer must contain the coordinates and weights in an SoA format
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points, float* input, int* output);

    /// @brief Constructs a container for the points allocated on the host using separate coordinate and weight buffers
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param coordinates The pre-allocated buffer containing the coordinates
    /// @param weights The pre-allocated buffer containing the weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The coordinates buffer must have a size of n_points * Ndim
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points, float* coordinates, float* weights, int* output);

    /// @brief Constructs a container for the points allocated on the host using multiple pre-allocated buffers
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param buffers The pre-allocated buffers to use for the points data
    template <concepts::queue TQueue, concepts::pointer... TBuffers>
      requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
    PointsHost(TQueue& queue, int32_t n_points, TBuffers... buffers);

    PointsHost(const PointsHost&) = delete;
    PointsHost& operator=(const PointsHost&) = delete;
    PointsHost(PointsHost&&) = default;
    PointsHost& operator=(PointsHost&&) = default;
    ~PointsHost() = default;

#ifdef CLUE_BUILD_DOXYGEN
    /// @brief Returns the number of points
    /// @return The number of points
    ALPAKA_FN_HOST int32_t size() const;
    /// @brief Returns the coordinates of the points for a specific dimension as a const span
    /// @param dim The dimension for which to get the coordinates
    /// @return A const span of the coordinates for the specified dimension
    ALPAKA_FN_HOST auto coords(size_t dim) const;
    /// @brief Returns the coordinates of the points for a specific dimension as a span
    /// @param dim The dimension for which to get the coordinates
    /// @return A span of the coordinates for the specified dimension
    ALPAKA_FN_HOST auto coords(size_t dim);
    /// @brief Returns the weights of the points as a const span
    /// @return A const span of the weights of the points
    ALPAKA_FN_HOST auto weights() const;
    /// @brief Returns the weights of the points as a span
    /// @return A span of the weights of the points
    ALPAKA_FN_HOST auto weights();
    /// @brief Returns the cluster indexes of the points as a const span
    /// @return A const span of the cluster indexes of the points
    ALPAKA_FN_HOST auto clusterIndexes() const;
    /// @brief Returns the cluster indexes of the points as a span
    /// @return A span of the cluster indexes of the points
    ALPAKA_FN_HOST auto clusterIndexes();
    /// @brief Indicates whether the points have been clustered
    /// @return True if the points have been clustered, false otherwise
    ALPAKA_FN_HOST auto clustered() const;
    /// @brief Returns the view of the points
    /// @return A const reference to the PointsView structure containing the points data
    ALPAKA_FN_HOST const auto& view() const;
    /// @brief Returns the view of the points
    /// @return A reference to the PointsView structure containing the points data
    ALPAKA_FN_HOST auto& view();
#endif

    /// @brief Returns the Point object at the specified index
    ///
    /// @param idx The index of the point to retrieve
    /// @return The Point object at the specified index
    Point operator[](std::size_t idx) const;

    /// @brief Teturns the cluster properties of the points
    ///
    /// @return The number of clusters reconstructed
    /// @note This value is lazily evaluated and cached upon the first call
    const auto& n_clusters();
    /// @brief Returns the associator mapping clusters to their associated points
    ///
    /// @return An host_associator mapping clusters to points
    /// @note This object is lazily evaluated and cached upon the first call
    const auto& clusters();
    /// @brief Returns a vector containing the sizes of each cluster
    ///
    /// @return A vector of containing the sizes of each cluster
    /// @note This vector is lazily evaluated and cached upon the first call
    const auto& cluster_sizes();
    /// @brief Returns the ClusterProperties object containing the properties of the clusters
    ///
    /// @return The ClusterProperties object
    /// @note This object is lazily evaluated and cached upon the first call
    const auto& cluster_properties();

  private:
    inline static constexpr std::size_t Ndim_ = Ndim;

    void mark_clustered() { m_clustered = true; }

    template <std::size_t _Ndim>
    friend class Clusterer;
    template <concepts::queue _TQueue, std::size_t _Ndim, concepts::device _TDev>
    friend void copyToHost(_TQueue& queue,
                           PointsHost<_Ndim>& h_points,
                           const PointsDevice<_Ndim, _TDev>& d_points);
    template <concepts::queue _TQueue, std::size_t _Ndim, concepts::device _TDev>
    friend void copyToDevice(_TQueue& queue,
                             PointsDevice<_Ndim, _TDev>& d_points,
                             const PointsHost<_Ndim>& h_points);
    friend struct internal::points_interface<PointsHost<Ndim>>;
    template <std::size_t NDim, concepts::queue TQueue>
    friend clue::PointsHost<NDim> read_output(TQueue& queue, const std::string& file_path);
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsHost.hpp"
#include "CLUEstering/data_structures/detail/Point.hpp"
