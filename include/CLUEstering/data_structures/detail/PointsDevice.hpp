
#pragma once

#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>

namespace clue {

  namespace soa::device {

    template <uint8_t Ndim>
    inline auto computeSoASize(int32_t n_points) {
      return ((Ndim + 3) * sizeof(float) + 3 * sizeof(int)) * n_points;
    }

    template <uint8_t Ndim>
    inline void partitionSoAView(PointsView& view, std::byte* buffer, int32_t n_points) {
      view.coords = reinterpret_cast<float*>(buffer);
      view.weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view.cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view.is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view.rho = reinterpret_cast<float*>(buffer + (Ndim + 3) * n_points * sizeof(float));
      view.delta = reinterpret_cast<float*>(buffer + (Ndim + 4) * n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(buffer + (Ndim + 5) * n_points * sizeof(float));
      view.n = n_points;
    }
    template <uint8_t Ndim>
    inline void partitionSoAView(PointsView& view,
                                 std::byte* alloc_buffer,
                                 std::byte* buffer,
                                 int32_t n_points) {
      view.coords = reinterpret_cast<float*>(buffer);
      view.weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view.cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view.is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view.rho = reinterpret_cast<float*>(alloc_buffer);
      view.delta = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view.n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    inline void partitionSoAView(PointsView& view,
                                 std::byte* alloc_buffer,
                                 int32_t n_points,
                                 TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);
      // TODO: is reinterpret_cast necessary?
      view.coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view.weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple));
      view.cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple));
      view.is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple));
      view.rho = reinterpret_cast<float*>(alloc_buffer);
      view.delta = reinterpret_cast<float*>(alloc_buffer + sizeof(float) * n_points);
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * sizeof(float) * n_points);
      view.n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    inline void partitionSoAView(PointsView& view,
                                 std::byte* alloc_buffer,
                                 int32_t n_points,
                                 TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);
      // TODO: is reinterpret_cast necessary?
      view.coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view.weight = reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Ndim * n_points);
      view.cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple));
      view.is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple) + n_points);
      view.rho = reinterpret_cast<float*>(alloc_buffer);
      view.delta = reinterpret_cast<float*>(alloc_buffer + sizeof(float) * n_points);
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * sizeof(float) * n_points);
      view.n = n_points;
    }

  }  // namespace soa::device

  template <uint8_t Ndim, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TDev>::PointsDevice(TQueue& queue, int32_t n_points)
      : m_buffer{make_device_buffer<std::byte[]>(queue,
                                                 soa::device::computeSoASize<Ndim>(n_points))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points);
  }

  template <uint8_t Ndim, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TDev>::PointsDevice(TQueue& queue,
                                                int32_t n_points,
                                                std::span<std::byte> buffer)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(float))},
        m_view{},
        m_size{n_points} {
    assert(buffer.size() == soa::device::computeSoASize<Ndim>(n_points));

    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), buffer.data(), n_points);
  }

  template <uint8_t Ndim, concepts::device TDev>
  template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
    requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
  inline PointsDevice<Ndim, TDev>::PointsDevice(TQueue& queue,
                                                int32_t n_points,
                                                TBuffers... buffers)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(float))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points, buffers...);
  }

  template <uint8_t Ndim, concepts::device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::rho() const {
    return std::span<const float>(m_view.rho, m_size);
  }
  template <uint8_t Ndim, concepts::device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::rho() {
    return std::span<float>(m_view.rho, m_size);
  }

  template <uint8_t Ndim, concepts::device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::delta() const {
    return std::span<const float>(m_view.delta, m_size);
  }
  template <uint8_t Ndim, concepts::device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::delta() {
    return std::span<float>(m_view.delta, m_size);
  }

  template <uint8_t Ndim, concepts::device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::nearestHigher() const {
    return std::span<const int>(m_view.nearest_higher, m_size);
  }
  template <uint8_t Ndim, concepts::device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::nearestHigher() {
    return std::span<int>(m_view.nearest_higher, m_size);
  }

}  // namespace clue
