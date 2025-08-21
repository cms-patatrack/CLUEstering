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
  private:
    std::optional<host_buffer<std::byte[]>> m_buffer;
    PointsView m_view;
    int32_t m_size;

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
    friend struct internal::points_interface<PointsHost<Ndim>>;
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsHost.hpp"
