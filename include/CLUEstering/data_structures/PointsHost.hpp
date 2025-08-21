/// @file PointsHost.hpp
/// @brief Provides the PointsHost class for managing points in host memory
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <optional>
#include <ranges>
#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  /// @brief The PointsHost class is a data structure that manages points in host memory.
  /// It provides methods to allocate, access, and manipulate points in host memory.
  ///
  /// @tparam Ndim The number of dimensions of the points to manage
  template <uint8_t Ndim>
  class PointsHost : public internal::points_interface<PointsHost<Ndim>> {
  public:
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points);

    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points, std::span<std::byte> buffer);

    template <concepts::queue TQueue, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsHost(TQueue& queue, int32_t n_points, TBuffers&&... buffers);

    template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsHost(TQueue& queue, int32_t n_points, TBuffers... buffers);

    PointsHost(const PointsHost&) = delete;
    PointsHost& operator=(const PointsHost&) = delete;
    PointsHost(PointsHost&&) = default;
    PointsHost& operator=(PointsHost&&) = default;
    ~PointsHost() = default;

  private:
    std::optional<host_buffer<std::byte[]>> m_buffer;
    PointsView m_view;
    int32_t m_size;

    inline static constexpr uint8_t Ndim_ = Ndim;

    friend struct internal::points_interface<PointsHost<Ndim>>;

    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToHost(_TQueue& queue,
                           PointsHost<_Ndim>& h_points,
                           const PointsDevice<_Ndim, _TDev>& d_points);
    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToDevice(_TQueue& queue,
                             PointsDevice<_Ndim, _TDev>& d_points,
                             const PointsHost<_Ndim>& h_points);
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsHost.hpp"
