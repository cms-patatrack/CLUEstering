
#pragma once

#include "CLUEstering/detail/concepts.hpp"

namespace clue {

  namespace concepts = detail::concepts;

  template <uint8_t Ndim>
  class PointsHost;
  template <uint8_t Ndim, concepts::device TDev>
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
    namespace concepts {

      template <typename T>
      concept contiguous_raw_data = std::is_array_v<T> || std::is_pointer_v<T>;

    }  // namespace concepts
  }  // namespace detail

  // TODO: implement for better cache use
  template <uint8_t Ndim>
  uint32_t computeAlignSoASize(uint32_t n_points);

  template <concepts::queue TQueue, uint8_t Ndim, concepts::device TDev>
  void copyToHost(TQueue queue,
                  PointsHost<Ndim>& h_points,
                  const PointsDevice<Ndim, TDev>& d_points) {
    alpaka::memcpy(queue,
                   make_host_view(h_points.m_view->cluster_index, h_points.size()),
                   make_device_view(
                       alpaka::getDev(queue), d_points.m_hostView->cluster_index, h_points.size()));
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view->is_seed, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_hostView->is_seed, h_points.size()));
  }
  template <concepts::queue TQueue, uint8_t Ndim, concepts::device TDev>
  void copyToDevice(TQueue queue,
                    PointsDevice<Ndim, TDev>& d_points,
                    const PointsHost<Ndim>& h_points) {
    alpaka::memcpy(queue,
                   make_device_view(
                       alpaka::getDev(queue), d_points.m_hostView->coords, Ndim * h_points.size()),
                   make_host_view(h_points.m_view->coords, Ndim * h_points.size()));
    alpaka::memcpy(
        queue,
        make_device_view(alpaka::getDev(queue), d_points.m_hostView->weight, h_points.size()),
        make_host_view(h_points.m_view->weight, h_points.size()));
  }

}  // namespace clue
