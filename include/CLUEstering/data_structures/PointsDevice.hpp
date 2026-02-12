/// @file PointsDevice.hpp
/// @brief Provides the PointsDevice class for managing points on a device
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <std::size_t Ndim, std::floating_point TData>
  class Clusterer;
  template <std::size_t Ndim, std::floating_point TData>
  class PointsHost;
  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  class PointsDevice;

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point THostInput,
            std::floating_point TDeviceInput,
            concepts::device TDev>
  void copyToHost(TQueue& queue,
                  PointsHost<Ndim, THostInput>& h_points,
                  const PointsDevice<Ndim, TDeviceInput, TDev>& d_points);

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TDeviceInput,
            concepts::device TDev,
            std::floating_point THostInput>
  void copyToDevice(TQueue& queue,
                    PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                    const PointsHost<Ndim, THostInput>& h_points);

  /// @brief The PointsDevice class is a data structure that manages points on a device.
  /// It provides methods to allocate, access, and manipulate points in device memory.
  ///
  /// @tparam Ndim The number of dimensions of the points to manage
  /// @tparam TData The data type for the point coordinates and weights
  /// @tparam TDev The device type to use for the allocation. Defaults to clue::Device.
  template <std::size_t Ndim, std::floating_point TData = float, concepts::device TDev = clue::Device>
  class PointsDevice : public internal::points_interface<PointsDevice<Ndim, TData, TDev>> {
  public:
    static_assert(std::is_same_v<TData, std::remove_reference_t<TData>>,
                  "Points' data must be a non-reference type");

    using element_type = TData;
    using value_type = std::remove_cv_t<TData>;

  private:
    device_buffer<TDev, std::byte[]> m_buffer;
    PointsView<Ndim, element_type> m_view;
    std::optional<std::size_t> m_nclusters;
    std::int32_t m_size;
    bool m_clustered = false;

  public:
    /// @brief Construct a PointsDevice object
    ///
    /// @param queue The queue to use for the device operations
    /// @param n_points The number of points to allocate
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, std::int32_t n_points);

    /// @brief Construct a PointsDevice object with a pre-allocated buffer
    ///
    /// @param queue The queue to use for the device operations
    /// @param n_points The number of points to allocate
    /// @param buffer The buffer to use for the points
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, std::int32_t n_points, std::span<std::byte> buffer);

    /// @brief Constructs a container for the points allocated on the device using interleaved data
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param input_buffer The pre-allocated buffer containing interleaved coordinates and weights
    /// @param output_buffer The pre-allocated buffer to store the cluster indexes
    /// @note The input buffer must contain the coordinates and weights in an SoA format
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue,
                 std::int32_t n_points,
                 std::span<element_type> input,
                 std::span<int> output);

    /// @brief Constructs a container for the points allocated on the device using separate coordinate and weight buffers
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param coordinates The pre-allocated buffer containing the coordinates
    /// @param weights The pre-allocated buffer containing the weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The coordinates buffer must have a size of n_points * Ndim
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue,
                 std::int32_t n_points,
                 std::span<element_type> coordinates,
                 std::span<element_type> weights,
                 std::span<int> output);

    /// @brief Constructs a container for the points allocated on the device using interleaved data
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param input_buffer The pre-allocated buffer containing interleaved coordinates and weights
    /// @param output_buffer The pre-allocated buffer to store the cluster indexes
    /// @note The input buffer must contain the coordinates and weights in an SoA format
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, std::int32_t n_points, element_type* input, int* output);

    /// @brief Constructs a container for the points allocated on the device using separate coordinate and weight buffers
    ///
    /// @param queue The queue to use for memory allocation
    /// @param n_points The number of points
    /// @param coordinates The pre-allocated buffer containing the coordinates
    /// @param weights The pre-allocated buffer containing the weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The coordinates buffer must have a size of n_points * Ndim
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue,
                 std::int32_t n_points,
                 element_type* coordinates,
                 element_type* weights,
                 int* output);

    /// @brief Construct a PointsDevice object with a pre-allocated buffer
    ///
    /// @param queue The queue to use for the device operations
    /// @param n_points The number of points to allocate
    /// @param buffers The buffers to use for the points
    template <concepts::queue TQueue, concepts::pointer... TBuffers>
      requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
    PointsDevice(TQueue& queue, std::int32_t n_points, TBuffers... buffers);

    PointsDevice(const PointsDevice&) = delete;
    PointsDevice& operator=(const PointsDevice&) = delete;
    PointsDevice(PointsDevice&&) = default;
    PointsDevice& operator=(PointsDevice&&) = default;
    ~PointsDevice() = default;

#ifdef CLUE_BUILD_DOXYGEN
    /// @brief Returns the number of points
    /// @return The number of points
    ALPAKA_FN_HOST auto size() const;
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

    /// @brief Teturns the cluster properties of the points
    ///
    /// @return The number of clusters reconstructed
    /// @note This value is lazily evaluated and cached upon the first call
    ALPAKA_FN_HOST const auto& n_clusters();

  private:
    inline static constexpr std::size_t Ndim_ = Ndim;

    void mark_clustered() { m_clustered = true; }

#ifndef CLUE_BUILD_DOXYGEN
    friend class Clusterer<Ndim, std::remove_cv_t<TData>>;

    template <concepts::queue TQueue,
              std::size_t N,
              std::floating_point THostInput,
              std::floating_point TDeviceInput,
              concepts::device Dev>
    friend void copyToHost(TQueue& queue,
                           PointsHost<N, THostInput>& h_points,
                           const PointsDevice<N, TDeviceInput, Dev>& d_points);

    template <concepts::queue TQueue,
              std::size_t N,
              std::floating_point TDeviceInput,
              concepts::device Dev,
              std::floating_point THostInput>
    friend void copyToDevice(TQueue& queue,
                             PointsDevice<N, TDeviceInput, Dev>& d_points,
                             const PointsHost<N, THostInput>& h_points);

    friend struct internal::points_interface<PointsDevice<Ndim, TData, TDev>>;
#endif
  };

  template <std::size_t Ndim, std::floating_point TData = float>
  using ConstPointsDevice = PointsDevice<Ndim, std::add_const_t<TData>>;

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsDevice.hpp"
