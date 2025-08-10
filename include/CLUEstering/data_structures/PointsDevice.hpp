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

  namespace concepts = detail::concepts;

  /// @brief The PointsDevice class is a data structure that manages points on a device.
  /// It provides methods to allocate, access, and manipulate points in device memory.
  ///
  /// @tparam Ndim The number of dimensions of the points to manage
  /// @tparam TDev The device type to use for the allocation. Defaults to clue::Device.
  template <uint8_t Ndim, concepts::device TDev = clue::Device>
  class PointsDevice {
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

    /// @brief Get the size of the points
    ///
    /// @return The number of points allocated
    ALPAKA_FN_HOST_ACC int32_t size() const;

    /// @brief Get the coordinates for a specific dimension
    ///
    /// @param dim The dimension to get the coordinates for
    /// @return A constant span of coordinates for the specified dimension
    ALPAKA_FN_HOST auto coords(size_t dim) const;
    /// @brief Get the coordinates for a specific dimension
    ///
    /// @param dim The dimension to get the coordinates for
    /// @return A span of coordinates for the specified dimension
    ALPAKA_FN_HOST auto coords(size_t dim);

    /// @brief Get the weights of all the points
    ///
    /// @return A constant span of weights for all the points
    ALPAKA_FN_HOST auto weight() const;
    /// @brief Get the weights of all the points
    ///
    /// @return A span of weights for all the points
    ALPAKA_FN_HOST auto weight();

    /// @brief Get the weighted density values of all the points
    ///
    /// @return A constant span of density values for all the points
    ALPAKA_FN_HOST auto rho() const;
    /// @brief Get the weighted density values of all the points
    ///
    /// @return A span of density values for all the points
    ALPAKA_FN_HOST auto rho();

    ALPAKA_FN_HOST auto delta() const;
    ALPAKA_FN_HOST auto delta();

    ALPAKA_FN_HOST auto nearestHigher() const;
    ALPAKA_FN_HOST auto nearestHigher();

    ALPAKA_FN_HOST auto clusterIndex() const;
    ALPAKA_FN_HOST auto clusterIndex();

    ALPAKA_FN_HOST auto isSeed() const;
    ALPAKA_FN_HOST auto isSeed();

    ALPAKA_FN_HOST const PointsView* view() const;
    ALPAKA_FN_HOST PointsView* view();

    /// @brief
    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToHost(_TQueue& queue,
                           PointsHost<_Ndim>& h_points,
                           const PointsDevice<_Ndim, _TDev>& d_points);
    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToDevice(_TQueue& queue,
                             PointsDevice<_Ndim, _TDev>& d_points,
                             const PointsHost<_Ndim>& h_points);

  private:
    device_buffer<TDev, std::byte[]> m_buffer;
    device_buffer<TDev, PointsView> m_view;
    host_buffer<PointsView> m_hostView;
    int32_t m_size;
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsDevice.hpp"
