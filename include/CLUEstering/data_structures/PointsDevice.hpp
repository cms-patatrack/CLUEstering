
#pragma once

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>

namespace clue {

  namespace concepts = detail::concepts;

  namespace soa::device {

    template <uint8_t Ndim>
    int32_t computeSoASize(int32_t n_points);

    template <uint8_t Ndim>
    void partitionSoAView(PointsView* view, std::byte* buffer, int32_t n_points);
    template <uint8_t Ndim>
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          std::byte* buffer,
                          int32_t n_points);
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          int32_t n_points,
                          TBuffers... buffer);
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    void partitionSoAView(PointsView* view,
                          std::byte* alloc_buffer,
                          int32_t n_points,
                          TBuffers... buffers);

  }  // namespace soa::device

  template <uint8_t Ndim, concepts::device TDev>
  class PointsDevice {
  public:
    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, int32_t n_points);

    template <concepts::queue TQueue>
    PointsDevice(TQueue& queue, int32_t n_points, std::span<std::byte> buffer);

    template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsDevice(TQueue& queue, int32_t n_points, TBuffers... buffers);

    PointsDevice(const PointsDevice&) = delete;
    PointsDevice& operator=(const PointsDevice&) = delete;
    PointsDevice(PointsDevice&&) = default;
    PointsDevice& operator=(PointsDevice&&) = default;
    ~PointsDevice() = default;

    ALPAKA_FN_HOST_ACC int32_t size() const;

    ALPAKA_FN_HOST auto coords(size_t dim) const;
    ALPAKA_FN_HOST auto coords(size_t dim);

    ALPAKA_FN_HOST auto weight() const;
    ALPAKA_FN_HOST auto weight();

    ALPAKA_FN_HOST auto rho() const;
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
