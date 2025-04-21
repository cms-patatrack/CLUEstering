
#pragma once

#include "../AlpakaCore/alpakaMemory.hpp"
#include "Common.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <utility>

namespace clue {

  namespace soa::device {
    template <uint8_t Ndim>
    uint32_t computeSoASize(uint32_t n_points) {
      return (Ndim + 3) * n_points * sizeof(float) + 3 * n_points * sizeof(int) +
             n_points * sizeof(uint8_t);
    }

    template <uint8_t Ndim>
    void partitionSoAView(PointsView* view, std::byte* buffer, uint32_t n_points) {
      view->coords = reinterpret_cast<float*>(buffer);
      view->weight = reinterpret_cast<float*>(buffer + Ndim * n_points);
      view->cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points);
      view->is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points);
      view->rho = reinterpret_cast<float*>(buffer + (Ndim + 4) * n_points);
      view->delta = reinterpret_cast<float*>(buffer + (Ndim + 5) * n_points);
      view->nearest_higher = reinterpret_cast<int*>(buffer + (Ndim + 6) * n_points);
      view->n = n_points;
    }
    template <uint8_t Ndim, detail::ArrayOrPtr... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          TBuffers... buffer,
                          uint32_t n_points) {
      auto buffers_tuple = std::make_tuple(buffer...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple));
      view->cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple));
      view->rho = reinterpret_cast<float*>(alloc_buffer);
      view->delta = reinterpret_cast<float*>(alloc_buffer + sizeof(float) * n_points);
      view->nearest_higher =
          reinterpret_cast<int*>(alloc_buffer + 2 * sizeof(float) * n_points);
      view->n = n_points;
    }
    template <uint8_t Ndim, detail::ArrayOrPtr... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          TBuffers... buffers,
                          uint32_t n_points) {
      auto buffers_tuple = std::make_tuple(buffers...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight =
          reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Ndim * n_points);
      view->cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple) + n_points);
      view->rho = reinterpret_cast<float*>(alloc_buffer);
      view->delta = reinterpret_cast<float*>(alloc_buffer + sizeof(float) * n_points);
      view->nearest_higher =
          reinterpret_cast<int*>(alloc_buffer + 2 * sizeof(float) * n_points);
      view->n = n_points;
    }
  }  // namespace soa::device

  template <uint8_t Ndim, typename TDev>
    requires alpaka::isDevice<TDev>
  class PointsDevice {
  public:
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    PointsDevice(TQueue queue, uint32_t n_points)
        : m_buffer{make_device_buffer<std::byte[]>(
              queue, soa::device::computeSoASize<Ndim>(n_points))},
          m_view{make_device_buffer<PointsView>(queue)},
          m_size{n_points} {
      auto h_view = make_host_buffer<PointsView>(queue);
      soa::device::partitionSoAView<Ndim>(h_view.data(), m_buffer->data(), n_points);
      alpaka::memcpy(queue, m_view, h_view);
    }
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    PointsDevice(TQueue queue, uint32_t n_points, std::span<std::byte> buffer)
        : m_view{make_device_buffer<PointsView>(queue)}, m_size{n_points} {
      assert(buffer.size() == soa::device::computeSoASize<Ndim>(n_points));

      auto h_view = make_host_buffer<PointsView>(queue);
      soa::device::partitionSoAView<Ndim>(h_view.data(), m_buffer->data(), n_points);
      alpaka::memcpy(queue, m_view, h_view);
    }
    template <typename TQueue, detail::ArrayOrPtr... TBuffers>
      requires alpaka::isQueue<TQueue> &&
                   (sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsDevice(TQueue queue, uint32_t n_points, TBuffers... buffers)
        : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(float))},
          m_view{make_device_buffer<PointsView>(queue)},
          m_size{n_points} {
      auto h_view = make_host_buffer<PointsView>(queue);
      soa::device::partitionSoAView<Ndim>(
          h_view.data(), m_buffer.data(), buffers..., n_points);
      alpaka::memcpy(queue, m_view, h_view);
    }

    PointsDevice(const PointsDevice&) = delete;
    PointsDevice& operator=(const PointsDevice&) = delete;
    PointsDevice(PointsDevice&&) = default;
    PointsDevice& operator=(PointsDevice&&) = default;
    ~PointsDevice() = default;

    ALPAKA_FN_HOST_ACC uint32_t size() const { return m_view->n; }

    ALPAKA_FN_HOST PointsView* view() { return m_view.data(); }
    ALPAKA_FN_HOST const PointsView* view() const { return m_view.data(); }

  private:
    std::optional<device_buffer<TDev, std::byte[]>> m_buffer;
    device_buffer<TDev, PointsView> m_view;
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
