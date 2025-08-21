/// @file PointsDevice.hpp
/// @brief Provides the PointsDevice class for managing points on a device
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <optional>
#include <ranges>
#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  /// @brief The PointsDevice class is a data structure that manages points on a device.
  /// It provides methods to allocate, access, and manipulate points in device memory.
  ///
  /// @tparam Ndim The number of dimensions of the points to manage
  /// @tparam TDev The device type to use for the allocation. Defaults to clue::Device.
  template <uint8_t Ndim, concepts::device TDev = clue::Device>
  class PointsDevice : public internal::points_interface<PointsDevice<Ndim, TDev>> {
  private:
    device_buffer<TDev, std::byte[]> m_buffer;
    PointsView m_view;
    int32_t m_size;

  public:
    /// @brief Construct a PointsDevice object
    ///
    /// @param queue The queue to use for the device operations
    /// @param n_points The number of points to allocate
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, int32_t n_points);

    /// @brief Construct a PointsDevice object with a pre-allocated buffer
    ///
    /// @param queue The queue to use for the device operations
    /// @param n_points The number of points to allocate
    /// @param buffer The buffer to use for the points
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, int32_t n_points, std::span<std::byte> buffer);

    /// @brief Construct a PointsDevice object with a pre-allocated buffer
    ///
    /// @param queue The queue to use for the device operations
    /// @param n_points The number of points to allocate
    /// @param buffers The buffers to use for the points
    template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsDevice(TQueue& queue, int32_t n_points, TBuffers... buffers);

    PointsDevice(const PointsDevice&) = delete;
    PointsDevice& operator=(const PointsDevice&) = delete;
    PointsDevice(PointsDevice&&) = default;
    PointsDevice& operator=(PointsDevice&&) = default;
    ~PointsDevice() = default;

#ifdef CLUE_BUILD_DOXYGEN
    /// @brief Returns the number of points
    /// @return The number of points
    ALPAKA_FN_HOST int32_t size() const;
    /// @brief Returns the coordinates of the points as a const span
    /// @return A const span of the coordinates of the points
    ALPAKA_FN_HOST auto coords() const;
    /// @brief Returns the coordinates of the points as a span
    /// @return A span of the coordinates of the points
    ALPAKA_FN_HOST auto coords();
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
    /// @brief Returns the seed status of the points as a const span
    /// @return A const span indicating whether each point is a seed
    ALPAKA_FN_HOST auto isSeed() const;
    /// @brief Returns the seed status of the points as a span
    /// @return A span indicating whether each point is a seed
    ALPAKA_FN_HOST auto isSeed();
    /// @brief Returns the view of the points
    /// @return A const reference to the PointsView structure containing the points data
    ALPAKA_FN_HOST const auto& view() const;
    /// @brief Returns the view of the points
    /// @return A reference to the PointsView structure containing the points data
    ALPAKA_FN_HOST auto& view();
#endif

    ALPAKA_FN_HOST auto rho() const;
    ALPAKA_FN_HOST auto rho();

    ALPAKA_FN_HOST auto delta() const;
    ALPAKA_FN_HOST auto delta();

    ALPAKA_FN_HOST auto nearestHigher() const;
    ALPAKA_FN_HOST auto nearestHigher();

  private:
    inline static constexpr uint8_t Ndim_ = Ndim;

    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToHost(_TQueue& queue,
                           PointsHost<_Ndim>& h_points,
                           const PointsDevice<_Ndim, _TDev>& d_points);
    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToDevice(_TQueue& queue,
                             PointsDevice<_Ndim, _TDev>& d_points,
                             const PointsHost<_Ndim>& h_points);
    friend struct internal::points_interface<PointsDevice<Ndim, TDev>>;
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsDevice.hpp"
