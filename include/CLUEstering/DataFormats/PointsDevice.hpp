
#pragma once

#include "CLUEstering/AlpakaCore/alpakaMemory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/DataFormats/Common.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>

namespace clue {

  namespace concepts = detail::concepts;

  namespace soa::device {

    template <uint8_t Ndim>
    int32_t computeSoASize(int32_t n_points) {
      return ((Ndim + 3) * sizeof(float) + 3 * sizeof(int)) * n_points;
    }

    template <uint8_t Ndim>
    void partitionSoAView(PointsView* view, std::byte* buffer, int32_t n_points) {
      view->coords = reinterpret_cast<float*>(buffer);
      view->weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view->cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view->is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view->rho = reinterpret_cast<float*>(buffer + (Ndim + 3) * n_points * sizeof(float));
      view->delta = reinterpret_cast<float*>(buffer + (Ndim + 4) * n_points * sizeof(float));
      view->nearest_higher = reinterpret_cast<int*>(buffer + (Ndim + 5) * n_points * sizeof(float));
      view->n = n_points;
    }
    template <uint8_t Ndim>
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          std::byte* buffer,
                          int32_t n_points) {
      view->coords = reinterpret_cast<float*>(buffer);
      view->weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view->cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view->is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view->rho = reinterpret_cast<float*>(alloc_buffer);
      view->delta = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view->nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view->n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          int32_t n_points,
                          TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple));
      view->cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple));
      view->rho = reinterpret_cast<float*>(alloc_buffer);
      view->delta = reinterpret_cast<float*>(alloc_buffer + sizeof(float) * n_points);
      view->nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * sizeof(float) * n_points);
      view->n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          int32_t n_points,
                          TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight = reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Ndim * n_points);
      view->cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple) + n_points);
      view->rho = reinterpret_cast<float*>(alloc_buffer);
      view->delta = reinterpret_cast<float*>(alloc_buffer + sizeof(float) * n_points);
      view->nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * sizeof(float) * n_points);
      view->n = n_points;
    }

  }  // namespace soa::device

  template <uint8_t Ndim, concepts::device TDev>
  class PointsDevice {
  public:
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, int32_t n_points)
        : m_buffer{make_device_buffer<std::byte[]>(queue,
                                                   soa::device::computeSoASize<Ndim>(n_points))},
          m_view{make_device_buffer<PointsView>(queue)},
          m_hostView{make_host_buffer<PointsView>(queue)},
          m_size{n_points} {
      soa::device::partitionSoAView<Ndim>(m_hostView.data(), m_buffer.data(), n_points);
      alpaka::memcpy(queue, m_view, m_hostView);
    }

    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, int32_t n_points, std::span<std::byte> buffer)
        : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(float))},
          m_view{make_device_buffer<PointsView>(queue)},
          m_hostView{make_host_buffer<PointsView>(queue)},
          m_size{n_points} {
      assert(buffer.size() == soa::device::computeSoASize<Ndim>(n_points));

      soa::device::partitionSoAView<Ndim>(
          m_hostView.data(), m_buffer.data(), buffer.data(), n_points);
      alpaka::memcpy(queue, m_view, m_hostView);
    }

    template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsDevice(TQueue& queue, int32_t n_points, TBuffers... buffers)
        : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(float))},
          m_view{make_device_buffer<PointsView>(queue)},
          m_hostView{make_host_buffer<PointsView>(queue)},
          m_size{n_points} {
      soa::device::partitionSoAView<Ndim>(m_hostView.data(), m_buffer.data(), n_points, buffers...);
      alpaka::memcpy(queue, m_view, m_hostView);
    }

    PointsDevice(const PointsDevice&) = delete;
    PointsDevice& operator=(const PointsDevice&) = delete;
    PointsDevice(PointsDevice&&) = default;
    PointsDevice& operator=(PointsDevice&&) = default;
    ~PointsDevice() = default;

    ALPAKA_FN_HOST_ACC int32_t size() const { return m_size; }

    ALPAKA_FN_HOST auto coords(size_t dim) const {
      assert(dim < Ndim);
      return std::span<const float>(m_hostView.data()->coords + dim * m_size, m_size);
    }
    ALPAKA_FN_HOST auto coords(size_t dim) {
      assert(dim < Ndim);
      return std::span<float>(m_hostView.data()->coords + dim * m_size, m_size);
    }

    ALPAKA_FN_HOST auto weight() const {
      return std::span<const float>(m_hostView.data()->weight, m_size);
    }
    ALPAKA_FN_HOST auto weight() { return std::span<float>(m_hostView.data()->weight, m_size); }

    ALPAKA_FN_HOST auto rho() const {
      return std::span<const float>(m_hostView.data()->rho, m_size);
    }
    ALPAKA_FN_HOST auto rho() { return std::span<float>(m_hostView.data()->rho, m_size); }

    ALPAKA_FN_HOST auto delta() const {
      return std::span<const float>(m_hostView.data()->delta, m_size);
    }
    ALPAKA_FN_HOST auto delta() { return std::span<float>(m_hostView.data()->delta, m_size); }

    ALPAKA_FN_HOST auto nearestHigher() const {
      return std::span<const int>(m_hostView.data()->nearest_higher, m_size);
    }
    ALPAKA_FN_HOST auto nearestHigher() {
      return std::span<int>(m_hostView.data()->nearest_higher, m_size);
    }

    ALPAKA_FN_HOST auto clusterIndex() const {
      return std::span<const int>(m_hostView.data()->cluster_index, m_size);
    }
    ALPAKA_FN_HOST auto clusterIndex() {
      return std::span<int>(m_hostView.data()->cluster_index, m_size);
    }

    ALPAKA_FN_HOST auto isSeed() const {
      return std::span<const int>(m_hostView.data()->is_seed, m_size);
    }
    ALPAKA_FN_HOST auto isSeed() { return std::span<int>(m_hostView.data()->is_seed, m_size); }

    ALPAKA_FN_HOST PointsView* view() { return m_view.data(); }
    ALPAKA_FN_HOST const PointsView* view() const { return m_view.data(); }

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
