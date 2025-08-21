
#pragma once

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>

namespace clue {

  namespace soa::host {

    // No need to allocate temporary buffers on the host
    template <uint8_t Ndim>
    inline auto computeSoASize(int32_t n_points) {
      return ((Ndim + 1) * sizeof(float) + 2 * sizeof(int)) * n_points;
    }

    template <uint8_t Ndim>
    inline void partitionSoAView(PointsView& view, std::byte* buffer, int32_t n_points) {
      view.coords = reinterpret_cast<float*>(buffer);
      view.weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view.cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view.is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view.n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    inline void partitionSoAView(PointsView& view, int32_t n_points, TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);
      // TODO: is reinterpret_cast necessary?
      view.coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view.weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple));
      view.cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple));
      view.is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple));
      view.n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    inline void partitionSoAView(PointsView& view, int32_t n_points, TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      // TODO: is reinterpret_cast necessary?
      view.coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view.weight = reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Ndim * n_points);
      view.cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple));
      view.is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple) + n_points);
      view.n = n_points;
    }
    template <uint8_t Ndim, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    inline void partitionSoAView(PointsView& view, int32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);
      // TODO: is reinterpret_cast necessary?
      view.coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data());
      view.weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple).data());
      view.cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple).data());
      view.is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple).data());
      view.n = n_points;
    }
    template <uint8_t Ndim, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    inline void partitionSoAView(PointsView& view, int32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);
      // TODO: is reinterpret_cast necessary?
      view.coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data());
      view.weight = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data() + Ndim * n_points);
      view.cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple).data());
      view.is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple).data() + n_points);
      view.n = n_points;
    }

  }  // namespace soa::host

  template <uint8_t Ndim>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points)
      : m_buffer{make_host_buffer<std::byte[]>(queue, soa::host::computeSoASize<Ndim>(n_points))},
        m_view{},
        m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, m_buffer->data(), n_points);
  }

  template <uint8_t Ndim>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points, std::span<std::byte> buffer)
      : m_view{}, m_size{n_points} {
    assert(buffer.size() == soa::host::computeSoASize<Ndim>(n_points));

    soa::host::partitionSoAView<Ndim>(m_view, buffer.data(), n_points);
  }

  template <uint8_t Ndim>
  template <concepts::queue TQueue, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points, TBuffers&&... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, std::forward<TBuffers>(buffers)...);
  }

  template <uint8_t Ndim>
  template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
    requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points, TBuffers... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, buffers...);
  }

}  // namespace clue
