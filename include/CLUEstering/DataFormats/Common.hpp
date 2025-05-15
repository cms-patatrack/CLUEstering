
#pragma once

namespace clue {

  template <uint8_t Ndim>
  class PointsHost;
  template <uint8_t Ndim, typename TDev>
    requires alpaka::isDevice<TDev>
  class PointsDevice;

  struct PointsView {
    float* coords;
    float* weight;
    int* cluster_index;
    int* is_seed;
    float* rho;
    float* delta;
    int* nearest_higher;
    uint32_t n;
  };

  namespace detail {

    template <typename T>
    concept ContiguousRange = requires(T&& t) {
      t.size();
      t.data();
    } && std::ranges::contiguous_range<T>;

    template <typename T>
    concept ArrayOrPtr = std::is_array_v<T> || std::is_pointer_v<T>;

  }  // namespace detail

  // TODO: implement for better cache use
  template <uint8_t Ndim>
  uint32_t computeAlignSoASize(uint32_t n_points);

  template <typename TQueue, uint8_t Ndim, typename TDev>
    requires alpaka::isQueue<TQueue> && alpaka::isDevice<TDev>
  void copyToHost(TQueue queue,
                  PointsHost<Ndim>& h_points,
                  const PointsDevice<Ndim, TDev>& d_points) {
    const auto copyExtent = 2 * h_points.size();
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view->cluster_index, copyExtent),
        make_device_view(alpaka::getDev(queue), d_points.m_hostView->cluster_index, copyExtent));
  }
  template <typename TQueue, uint8_t Ndim, typename TDev>
    requires alpaka::isQueue<TQueue> && alpaka::isDevice<TDev>
  void copyToDevice(TQueue queue,
                    PointsDevice<Ndim, TDev>& d_points,
                    const PointsHost<Ndim>& h_points) {
    const auto copyExtent = (Ndim + 1) * h_points.size();
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), d_points.m_hostView->coords, copyExtent),
                   make_host_view(h_points.m_view->coords, copyExtent));
  }

}  // namespace clue
