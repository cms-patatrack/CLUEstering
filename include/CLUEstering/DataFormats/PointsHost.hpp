
#pragma once

#include "../AlpakaCore/alpakaMemory.hpp"
#include "Common.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <utility>

namespace clue {

  namespace soa::host {
    // No need to allocate temporary buffers on the host
    template <uint8_t Ndim>
    uint32_t computeSoASize(uint32_t n_points) {
      return (Ndim + 1) * n_points * sizeof(float) + 2 * n_points * sizeof(int) +
             n_points * sizeof(uint8_t);
    }

    template <uint8_t Ndim>
    void partitionSoAView(PointsView* view, std::byte* buffer, uint32_t n_points) {
      view->coords = reinterpret_cast<float*>(buffer);
      view->weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view->cluster_index =
          reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view->is_seed =
          reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view->n = n_points;
    }
    template <uint8_t Ndim, detail::ArrayOrPtr... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    void partitionSoAView(PointsView* view, uint32_t n_points, TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple));
      view->cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple));
      view->n = n_points;
    }
    template <uint8_t Ndim, detail::ArrayOrPtr... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    void partitionSoAView(PointsView* view, uint32_t n_points, TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight =
          reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Ndim * n_points);
      view->cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple) + n_points);
      view->n = n_points;
    }
    template <uint8_t Ndim, detail::ContiguousRange... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    void partitionSoAView(PointsView* view, uint32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data());
      view->weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple).data());
      view->cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple).data());
      view->is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple).data());
      view->n = n_points;
    }
    template <uint8_t Ndim, detail::ContiguousRange... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    void partitionSoAView(PointsView* view, uint32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data());
      view->weight =
          reinterpret_cast<float*>(std::get<0>(buffers_tuple).data() + Ndim * n_points);
      view->cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple).data());
      view->is_seed =
          reinterpret_cast<int*>(std::get<1>(buffers_tuple).data() + n_points);
      view->n = n_points;
    }
  }  // namespace soa::host

  template <uint8_t Ndim>
  class PointsHost {
  public:
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    PointsHost(TQueue queue, uint32_t n_points)
        : m_buffer{make_host_buffer<std::byte[]>(
              queue, soa::host::computeSoASize<Ndim>(n_points))},
          m_view{make_host_buffer<PointsView>(queue)},
          m_size{n_points} {
      auto h_view = make_host_buffer<PointsView>(queue);
      soa::host::partitionSoAView<Ndim>(h_view.data(), m_buffer->data(), n_points);
      alpaka::memcpy(queue, m_view, h_view);
    }
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    PointsHost(TQueue queue, uint32_t n_points, std::span<std::byte> buffer)
        : m_view{make_host_buffer<PointsView>(queue)}, m_size{n_points} {
      assert(buffer.size() == soa::host::computeSoASize<Ndim>(n_points));

      auto h_view = make_host_buffer<PointsView>(queue);
      soa::host::partitionSoAView<Ndim>(h_view.data(), buffer.data(), n_points);
      alpaka::memcpy(queue, m_view, h_view);
    }
    template <typename TQueue, detail::ContiguousRange... TBuffers>
      requires alpaka::isQueue<TQueue> &&
                   (sizeof...(TBuffers) == 4 || sizeof...(TBuffers) == 2)
    PointsHost(TQueue queue, uint32_t n_points, TBuffers&&... buffers)
        : m_view{make_host_buffer<PointsView>(queue)}, m_size{n_points} {
      // assert(buffer.size() == soa::host::computeSoASize<Ndim>(n_points));

      auto h_view = make_host_buffer<PointsView>(queue);
      soa::host::partitionSoAView<Ndim>(
          h_view.data(), n_points, std::forward<TBuffers>(buffers)...);
      alpaka::memcpy(queue, m_view, h_view);
    }
    template <typename TQueue, detail::ArrayOrPtr... TBuffers>
      requires alpaka::isQueue<TQueue>
    PointsHost(TQueue queue, uint32_t n_points, TBuffers... buffers)
        : m_view{make_host_buffer<PointsView>(queue)}, m_size{n_points} {
      // assert(buffer.size() == soa::host::computeSoASizeHost<Ndim>(n_points));

      auto h_view = make_host_buffer<PointsView>(queue);
      soa::host::partitionSoAView<Ndim>(h_view.data(), n_points, buffers...);
      alpaka::memcpy(queue, m_view, h_view);
    }

    PointsHost(const PointsHost&) = delete;
    PointsHost& operator=(const PointsHost&) = delete;
    PointsHost(PointsHost&&) = default;
    PointsHost& operator=(PointsHost&&) = default;
    ~PointsHost() = default;

    ALPAKA_FN_HOST uint32_t size() const { return m_view->n; }

    ALPAKA_FN_HOST std::span<const float> coords() const {
      return std::span<const float>(m_view->coords,
                                    static_cast<std::size_t>(m_view->n * Ndim));
    }
    ALPAKA_FN_HOST std::span<float> coords() {
      return std::span<float>(m_view->coords, static_cast<std::size_t>(m_view->n * Ndim));
    }

    ALPAKA_FN_HOST std::span<const float> coords(size_t dim) const {
      return std::span<const float>(m_view->coords + dim * m_view->n,
                                    static_cast<std::size_t>(m_view->n * Ndim));
    }
    ALPAKA_FN_HOST std::span<float> coords(size_t dim) {
      return std::span<float>(m_view->coords + dim * m_view->n,
                              static_cast<std::size_t>(m_view->n * Ndim));
    }

    ALPAKA_FN_HOST std::span<const float> weights() const {
      return std::span<const float>(m_view->weight, static_cast<std::size_t>(m_view->n));
    }
    ALPAKA_FN_HOST std::span<float> weights() {
      return std::span<float>(m_view->weight, static_cast<std::size_t>(m_view->n));
    }

    ALPAKA_FN_HOST std::span<const int> clusterIndexes() const {
      return std::span<const int>(m_view->cluster_index,
                                  static_cast<std::size_t>(m_view->n));
    }
    ALPAKA_FN_HOST std::span<int> clusterIndexes() {
      return std::span<int>(m_view->cluster_index, static_cast<std::size_t>(m_view->n));
    }

    ALPAKA_FN_HOST std::span<const int> isSeed() const {
      return std::span<const int>(m_view->is_seed, static_cast<std::size_t>(m_view->n));
    }
    ALPAKA_FN_HOST std::span<int> isSeed() {
      return std::span<int>(m_view->is_seed, static_cast<std::size_t>(m_view->n));
    }
    // ALPAKA_FN_HOST std::span<const uint8_t, Ndim> wrapping() const {
    //   return std::span<const uint8_t, Ndim>(m_view->wrapping,
    //                                         static_cast<std::size_t>(m_view->n));
    // }

    ALPAKA_FN_HOST PointsView* view() { return m_view.data(); }
    ALPAKA_FN_HOST const PointsView* view() const { return m_view.data(); }

  private:
    std::optional<host_buffer<std::byte[]>> m_buffer;
    host_buffer<PointsView> m_view;
    uint32_t m_size;
  };

  // deduction guide for deducing device type from queue
  // template <uint8_t Ndim, typename TQueue>
  //   requires alpaka::isDevice<TDev>
  // PointsDevice(const TQueue& queue, uint32_t n_points)
  //     -> Points<Ndim, decltype(alpaka::getDev(queue))>;

  // template <uint8_t Ndim, typename TDev>
  //   requires alpaka::isDevice<TDev>
  // void copyToHost(PointsHost<Ndim>& h_points, Points<Ndim, TDev>& d_points) {}

}  // namespace clue
