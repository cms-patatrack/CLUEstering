
#pragma once

#include <alpaka/alpaka.hpp>
#include <span>

namespace clue {

  namespace detail {
    template <uint8_t Ndim>
    inline uint32_t computeSoASize(uint32_t n_points) {
      return (Ndim + 3) * n_points * sizeof(float) + 3 * n_points * sizeof(int) +
             n_points * sizeof(uint8_t);
    }

    template <uint8_t Ndim>
    inline void partitionSoAView(PointsView* view, uint32_t n_points) {
      view->coords = reinterpret_cast<float*>(m_buffer.data());
      view->weight = reinterpret_cast<float*>(m_buffer.data() + Ndim * n_points);
      view->cluster_index =
          reinterpret_cast<int*>(m_buffer.data() + (Ndim + 1) * n_points);
      view->is_seed = reinterpret_cast<int*>(m_buffer.data() + (Ndim + 2) * n_points);
      view->wrapping =
          reinterpret_cast<uint8_t*>(m_buffer.data() + (Ndim + 3) * n_points);
      view->rho = reinterpret_cast<float*>(m_buffer.data() + (Ndim + 4) * n_points);
      view->delta = reinterpret_cast<float*>(m_buffer.data() + (Ndim + 5) * n_points);
      view->nearest_higher =
          reinterpret_cast<int*>(m_buffer.data() + (Ndim + 6) * n_points);
      view->n = n_points;
    }
  }  // namespace detail

  struct PointsView {
    float* coords;
    float* weight;
    int* cluster_index;
    int* is_seed;
    uint8_t* wrapping;
    float* rho;
    float* delta;
    int* nearest_higher;
    uint32_t n;
  };

  template <uint8_t Ndim, typename TDev>
    requires alpaka::isDevice<TDev>
  class Points {
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    ALPAKA_FN_HOST Points(const TQueue& queue, uint32_t n_points)
        : m_buffer{make_device_buffer<std::byte[]>(
              queue, detail::computeSoASize<Ndim>(n_points))},
          m_view{make_device_buffer<PointsView>(queue)} {
      auto h_view = make_host_buffer<PointsView>(queue);
      detail::partitionSoAView<Ndim>(h_view.data(), n_points);
      alpaka::memcpy(queue, m_view, h_view);
    }
    ALPAKA_FN_HOST Points(std::span<std::byte> buffer, uint32_t n_points)
        : m_view{make_device_buffer<PointsView>(queue)} {
      assert(buffer.size() == detail::computeSoASize<Ndim>(n_points));

      auto h_view = make_host_buffer<PointsView>(queue);
      detail::partitionSoAView<Ndim>(h_view.data(), n_points);
      alpaka::memcpy(queue, m_view, h_view);
    }

    Points(const Points&) = delete;
    Points& operator=(const Points&) = delete;
    Points(Points&&) = default;
    Points& operator=(Points&&) = default;
    ~Points() = default;

    ALPAKA_FN_HOST_ACC uint32_t size() const { return m_view->n; }

    ALPAKA_FN_HOST_ACC std::span<const float> coords() const {
      return std::span<float>(m_view->coords, m_view->n * Ndim);
    }
    ALPAKA_FN_HOST_ACC std::span<const float> weights() const {
      return std::span<float>(m_view->weight, m_view->n);
    }
    ALPAKA_FN_HOST_ACC std::span<const int> clusterIndexes() const {
      return std::span<const int>(m_view->cluster_index, m_view->n);
    }
    ALPAKA_FN_HOST_ACC std::span<const int> isSeed() const {
      return std::span<const int>(m_view->is_seed, m_view->n);
    }
    ALPAKA_FN_HOST_ACC std::span<const uint8_t, Ndim> wrapping() const {
      return std::span<const uint8_t, Ndim>(m_view->wrapping, m_view->n);
    }

    ALPAKA_FN_HOST PointsView* view() { return m_view.data(); }

  private:
    device_buffer<TDev, std::byte[]> m_buffer;
    device_buffer<TDev, PointsView> m_view;
  };

  // deduction guide for deducing device type from queue
  // template <uint8_t Ndim, typename TQueue>
  //   requires alpaka::isDevice<TDev>
  // Points(const TQueue& queue, uint32_t n_points)
  //     -> Points<Ndim, decltype(alpaka::getDev(queue))>;

  template <uint8_t Ndim>
  using PointsHost = Points<Ndim, alpaka::DevCpu>;

  template <uint8_t Ndim, typename TDev>
    requires alpaka::isDevice<TDev>
  using PointsDevice = Points<Ndim, TDev>;

  template <uint8_t Ndim, typename TDev>
    requires alpaka::isDevice<TDev>
  void copyToHost(PointsHost<Ndim>& h_points, Poinst<Ndim, TDev>& d_points) {}

}  // namespace clue
