

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>

namespace clue {

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point THostInput,
            std::floating_point TDeviceInput,
            concepts::device TDev>
  inline void copyToHost(TQueue& queue,
                         PointsHost<Ndim, THostInput>& h_points,
                         const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
    alpaka::memcpy(
        queue,
        make_host_view(h_points.view().m_cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.view().m_cluster_index, h_points.size()));
    internal::points_interface<std::remove_cvref_t<decltype(h_points)>>::mark_clustered(h_points);
    alpaka::wait(queue);
  }

  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
  inline auto copyToHost(TQueue& queue, const PointsDevice<Ndim, TInput, TDev>& d_points) {
    PointsHost<Ndim, std::remove_cv_t<TInput>> h_points(queue, d_points.size());

    alpaka::memcpy(
        queue,
        make_host_view(h_points.view().m_cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.view().m_cluster_index, h_points.size()));
    internal::points_interface<std::remove_cvref_t<decltype(h_points)>>::mark_clustered(h_points);
    alpaka::wait(queue);

    return h_points;
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TDeviceInput,
            concepts::device TDev,
            std::floating_point THostInput>
  inline void copyToDevice(TQueue& queue,
                           PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                           const PointsHost<Ndim, THostInput>& h_points) {
    meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), d_points.view().m_coords[Dim], h_points.size()),
          make_host_view(h_points.view().m_coords[Dim], h_points.size()));
    });
    alpaka::memcpy(
        queue,
        make_device_view(alpaka::getDev(queue), d_points.view().m_weight, h_points.size()),
        make_host_view(h_points.view().m_weight, h_points.size()));
    if (h_points.view().has_uncertainty()) {
      using dev_value_t = std::remove_cv_t<TDeviceInput>;
      using PType = std::remove_cvref_t<decltype(d_points)>;
      auto& ubuf = internal::points_interface<PType>::uncertainty_buffer(d_points);
      ubuf = make_device_buffer<dev_value_t[]>(queue, h_points.size());
      alpaka::memcpy(queue,
                     make_device_view(alpaka::getDev(queue), ubuf->data(), h_points.size()),
                     make_host_view(h_points.view().m_density_uncertainty, h_points.size()));
      d_points.view().m_density_uncertainty = ubuf->data();
    }
    alpaka::wait(queue);
  }

  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
  inline auto copyToDevice(TQueue& queue, const PointsHost<Ndim, TInput>& h_points) {
    PointsDevice<Ndim, std::remove_cv_t<TInput>, TDev> d_points(queue, h_points.size());

    meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), d_points.view().m_coords[Dim], h_points.size()),
          make_host_view(h_points.view().m_coords[Dim], h_points.size()));
    });
    alpaka::memcpy(
        queue,
        make_device_view(alpaka::getDev(queue), d_points.view().m_weight, h_points.size()),
        make_host_view(h_points.view().m_weight, h_points.size()));
    if (h_points.view().has_uncertainty()) {
      using dev_value_t = std::remove_cv_t<TInput>;
      using PType = std::remove_cvref_t<decltype(d_points)>;
      auto& ubuf = internal::points_interface<PType>::uncertainty_buffer(d_points);
      ubuf = make_device_buffer<dev_value_t[]>(queue, h_points.size());
      alpaka::memcpy(queue,
                     make_device_view(alpaka::getDev(queue), ubuf->data(), h_points.size()),
                     make_host_view(h_points.view().m_density_uncertainty, h_points.size()));
      d_points.view().m_density_uncertainty = ubuf->data();
    }
    alpaka::wait(queue);

    return d_points;
  }

}  // namespace clue
