

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>

namespace clue {

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  inline void copyToHost(TQueue& queue,
                         PointsHost<Ndim>& h_points,
                         const PointsDevice<Ndim, TDev>& d_points) {
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view.cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
    h_points.mark_clustered();
  }

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  inline void copyToDevice(TQueue& queue,
                           PointsDevice<Ndim, TDev>& d_points,
                           const PointsHost<Ndim>& h_points) {
    meta::apply<Ndim>([&]<std::size_t Dim> {
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), d_points.m_view.coords[Dim], h_points.size()),
          make_host_view(h_points.m_view.coords[Dim], Ndim * h_points.size()));
    });
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));
  }

}  // namespace clue
