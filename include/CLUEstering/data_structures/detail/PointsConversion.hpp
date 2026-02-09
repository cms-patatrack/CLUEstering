

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>

namespace clue {

  // TODO: separate template parameters for input and output points, and add static asserts for constness
  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  inline void copyToHost(TQueue& queue,
                         PointsHost<Ndim, TData>& h_points,
                         const PointsDevice<Ndim, TData, TDev>& d_points) {
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view.cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
    h_points.mark_clustered();
  }

  // TODO: separate template parameters for input and output points, and add static asserts for constness
  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  inline auto copyToHost(TQueue& queue, const PointsDevice<Ndim, TData, TDev>& d_points) {
    PointsHost<Ndim, TData> h_points(queue, d_points.size());

    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view.cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
    h_points.mark_clustered();

    return h_points;
  }

  // TODO: separate template parameters for input and output points, and add static asserts for constness
  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  inline void copyToDevice(TQueue& queue,
                           PointsDevice<Ndim, TData, TDev>& d_points,
                           const PointsHost<Ndim, TData>& h_points) {
    meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), d_points.m_view.coords[Dim], h_points.size()),
          make_host_view(h_points.m_view.coords[Dim], h_points.size()));
    });
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));
  }

  // TODO: separate template parameters for input and output points, and add static asserts for constness
  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  inline auto copyToDevice(TQueue& queue, const PointsHost<Ndim, TData>& h_points) {
    PointsDevice<Ndim, TData, TDev> d_points(queue, h_points.size());

    meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), d_points.m_view.coords[Dim], h_points.size()),
          make_host_view(h_points.m_view.coords[Dim], h_points.size()));
    });
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));

    return d_points;
  }

}  // namespace clue
