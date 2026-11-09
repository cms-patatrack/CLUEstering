

#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/alpaka/buffer.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

CLUE_NAMESPACE_BEGIN

namespace internal {

  // Allocates the device buffer for the distances to the nearest higher, if the host points
  // requested the clustering details to be stored
  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TDeviceInput,
            concepts::device TDev,
            std::floating_point THostInput,
            concepts::allocator TAllocator>
  inline void setup_details_buffers(TQueue& queue,
                                    PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                                    const PointsHost<Ndim, THostInput>& h_points,
                                    const TAllocator& allocator) {
    if (!h_points.has_details())
      return;

    using dev_value_t = std::remove_cv_t<TDeviceInput>;
    using PType = std::remove_cvref_t<decltype(d_points)>;
    auto& dbuf = points_interface<PType>::delta_buffer(d_points);
    if (!dbuf.has_value())
      dbuf = alloc_device_buffer<dev_value_t>(
          queue, static_cast<std::size_t>(d_points.size()), allocator);
    d_points.view().m_delta = dbuf->data();
  }

  // Checks that the device points store the clustering details requested by the host points.
  // It must be called before enqueuing any copy, so that no copy is left pending if it throws
  template <std::size_t Ndim,
            std::floating_point THostInput,
            std::floating_point TDeviceInput,
            concepts::device TDev>
  inline void check_details_available(const PointsHost<Ndim, THostInput>& h_points,
                                      const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
    if (h_points.has_details() && !d_points.view().has_delta()) {
      throw std::invalid_argument(
          "The host points request the clustering details, but the device points do not store "
          "them. Call store_details on the host points before copying them to the device.");
    }
  }

  // Enqueues the copy of the densities, the distances to the nearest higher and the nearest
  // highers from the device to the host, if the host points requested them
  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point THostInput,
            std::floating_point TDeviceInput,
            concepts::device TDev>
  inline void copy_details_to_host(TQueue& queue,
                                   PointsHost<Ndim, THostInput>& h_points,
                                   const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
    if (!h_points.has_details())
      return;

    auto& h_view = h_points.view();
    const auto& d_view = d_points.view();
    const auto n_points = h_points.size();
    auto copy = [&](auto* dst, const auto* src) {
      if (dst != src)
        alpaka::memcpy(queue,
                       alpaka::createView(host, dst, n_points),
                       alpaka::createView(queue, src, n_points));
    };
    copy(h_view.m_rho, d_view.m_rho);
    copy(h_view.m_delta, d_view.m_delta);
    copy(h_view.m_nearest_higher, d_view.m_nearest_higher);
  }

}  // namespace internal

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point THostInput,
          std::floating_point TDeviceInput,
          concepts::device TDev>
inline void copyToHost(TQueue& queue,
                       PointsHost<Ndim, THostInput>& h_points,
                       const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
  internal::check_details_available(h_points, d_points);
  alpaka::memcpy(queue,
                 alpaka::createView(host, h_points.view().m_cluster_index, h_points.size()),
                 alpaka::createView(queue, d_points.view().m_cluster_index, h_points.size()));
  internal::copy_details_to_host(queue, h_points, d_points);
  alpaka::wait(queue);
  internal::points_interface<std::remove_cvref_t<decltype(h_points)>>::mark_clustered(h_points);
}

template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
inline auto copyToHost(TQueue& queue, const PointsDevice<Ndim, TInput, TDev>& d_points) {
  PointsHost<Ndim, std::remove_cv_t<TInput>> h_points(queue, d_points.size());
  if (d_points.view().has_delta())
    h_points.store_details();

  alpaka::memcpy(queue,
                 alpaka::createView(host, h_points.view().m_cluster_index, h_points.size()),
                 alpaka::createView(queue, d_points.view().m_cluster_index, h_points.size()));
  internal::copy_details_to_host(queue, h_points, d_points);
  alpaka::wait(queue);
  internal::points_interface<std::remove_cvref_t<decltype(h_points)>>::mark_clustered(h_points);

  return h_points;
}

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point THostInput,
          std::floating_point TDeviceInput,
          concepts::device TDev>
void copyToHostAsync(TQueue& queue,
                     PointsHost<Ndim, THostInput>& h_points,
                     const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
  internal::check_details_available(h_points, d_points);
  alpaka::memcpy(queue,
                 alpaka::createView(host, h_points.view().m_cluster_index, h_points.size()),
                 alpaka::createView(queue, d_points.view().m_cluster_index, h_points.size()));
  internal::copy_details_to_host(queue, h_points, d_points);
  internal::points_interface<std::remove_cvref_t<decltype(h_points)>>::mark_clustered(h_points);
}

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point TDeviceInput,
          concepts::device TDev,
          std::floating_point THostInput>
inline void copyToDevice(TQueue& queue,
                         PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                         const PointsHost<Ndim, THostInput>& h_points) {
  copyToDevice(queue, d_points, h_points, DefaultAllocator{});
}

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point TDeviceInput,
          concepts::device TDev,
          std::floating_point THostInput,
          concepts::allocator TAllocator>
inline void copyToDevice(TQueue& queue,
                         PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                         const PointsHost<Ndim, THostInput>& h_points,
                         const TAllocator& allocator) {
  meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
    alpaka::memcpy(queue,
                   alpaka::createView(queue, d_points.view().m_coords[Dim], h_points.size()),
                   alpaka::createView(host, h_points.view().m_coords[Dim], h_points.size()));
  });
  alpaka::memcpy(queue,
                 alpaka::createView(queue, d_points.view().m_weight, h_points.size()),
                 alpaka::createView(host, h_points.view().m_weight, h_points.size()));
  if (h_points.view().has_uncertainty()) {
    using dev_value_t = std::remove_cv_t<TDeviceInput>;
    using PType = std::remove_cvref_t<decltype(d_points)>;
    auto& ubuf = internal::points_interface<PType>::uncertainty_buffer(d_points);
    ubuf = internal::alloc_device_buffer<dev_value_t>(
        queue, static_cast<std::size_t>(h_points.size()), allocator);
    alpaka::memcpy(
        queue,
        alpaka::createView(queue, ubuf->data(), h_points.size()),
        alpaka::createView(host, h_points.view().m_density_uncertainty, h_points.size()));
    d_points.view().m_density_uncertainty = ubuf->data();
  }
  meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
    if (h_points.view().has_sigma(Dim)) {
      using dev_value_t = std::remove_cv_t<TDeviceInput>;
      using PType = std::remove_cvref_t<decltype(d_points)>;
      auto& cbufs = internal::points_interface<PType>::sigma_buffers(d_points);
      cbufs[Dim] = internal::alloc_device_buffer<dev_value_t>(
          queue, static_cast<std::size_t>(h_points.size()), allocator);
      alpaka::memcpy(queue,
                     alpaka::createView(queue, cbufs[Dim]->data(), h_points.size()),
                     alpaka::createView(host, h_points.view().m_sigmas[Dim], h_points.size()));
      d_points.view().m_sigmas[Dim] = cbufs[Dim]->data();
    }
  });
  if (h_points.view().has_tags()) {
    using PType = std::remove_cvref_t<decltype(d_points)>;
    auto& tbuf = internal::points_interface<PType>::tags_buffer(d_points);
    tbuf = internal::alloc_device_buffer<std::uint32_t>(
        queue, static_cast<std::size_t>(h_points.size()), allocator);
    alpaka::memcpy(queue,
                   alpaka::createView(queue, tbuf->data(), h_points.size()),
                   alpaka::createView(host, h_points.view().m_tags, h_points.size()));
    d_points.view().m_tags = tbuf->data();
  }
  internal::setup_details_buffers(queue, d_points, h_points, allocator);
  alpaka::wait(queue);
}

template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
inline auto copyToDevice(TQueue& queue, const PointsHost<Ndim, TInput>& h_points) {
  return copyToDevice<TQueue, Ndim, TInput, TDev>(queue, h_points, DefaultAllocator{});
}

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point TInput,
          concepts::device TDev,
          concepts::allocator TAllocator>
inline auto copyToDevice(TQueue& queue,
                         const PointsHost<Ndim, TInput>& h_points,
                         const TAllocator& allocator) {
  PointsDevice<Ndim, std::remove_cv_t<TInput>, TDev> d_points(queue, h_points.size(), allocator);

  meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
    alpaka::memcpy(queue,
                   alpaka::createView(queue, d_points.view().m_coords[Dim], h_points.size()),
                   alpaka::createView(host, h_points.view().m_coords[Dim], h_points.size()));
  });
  alpaka::memcpy(queue,
                 alpaka::createView(queue, d_points.view().m_weight, h_points.size()),
                 alpaka::createView(host, h_points.view().m_weight, h_points.size()));
  if (h_points.view().has_uncertainty()) {
    using dev_value_t = std::remove_cv_t<TInput>;
    using PType = std::remove_cvref_t<decltype(d_points)>;
    auto& ubuf = internal::points_interface<PType>::uncertainty_buffer(d_points);
    ubuf = internal::alloc_device_buffer<dev_value_t>(
        queue, static_cast<std::size_t>(h_points.size()), allocator);
    alpaka::memcpy(
        queue,
        alpaka::createView(queue, ubuf->data(), h_points.size()),
        alpaka::createView(host, h_points.view().m_density_uncertainty, h_points.size()));
    d_points.view().m_density_uncertainty = ubuf->data();
  }
  meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
    if (h_points.view().has_sigma(Dim)) {
      using dev_value_t = std::remove_cv_t<TInput>;
      using PType = std::remove_cvref_t<decltype(d_points)>;
      auto& cbufs = internal::points_interface<PType>::sigma_buffers(d_points);
      cbufs[Dim] = internal::alloc_device_buffer<dev_value_t>(
          queue, static_cast<std::size_t>(h_points.size()), allocator);
      alpaka::memcpy(queue,
                     alpaka::createView(queue, cbufs[Dim]->data(), h_points.size()),
                     alpaka::createView(host, h_points.view().m_sigmas[Dim], h_points.size()));
      d_points.view().m_sigmas[Dim] = cbufs[Dim]->data();
    }
  });
  if (h_points.view().has_tags()) {
    using PType = std::remove_cvref_t<decltype(d_points)>;
    auto& tbuf = internal::points_interface<PType>::tags_buffer(d_points);
    tbuf = internal::alloc_device_buffer<std::uint32_t>(
        queue, static_cast<std::size_t>(h_points.size()), allocator);
    alpaka::memcpy(queue,
                   alpaka::createView(queue, tbuf->data(), h_points.size()),
                   alpaka::createView(host, h_points.view().m_tags, h_points.size()));
    d_points.view().m_tags = tbuf->data();
  }
  internal::setup_details_buffers(queue, d_points, h_points, allocator);
  alpaka::wait(queue);

  return d_points;
}

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point TDeviceInput,
          concepts::device TDev,
          std::floating_point THostInput>
inline void copyToDeviceAsync(TQueue& queue,
                              PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                              const PointsHost<Ndim, THostInput>& h_points) {
  copyToDeviceAsync(queue, d_points, h_points, DefaultAllocator{});
}

template <concepts::queue TQueue,
          std::size_t Ndim,
          std::floating_point TDeviceInput,
          concepts::device TDev,
          std::floating_point THostInput,
          concepts::allocator TAllocator>
inline void copyToDeviceAsync(TQueue& queue,
                              PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                              const PointsHost<Ndim, THostInput>& h_points,
                              const TAllocator& allocator) {
  meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
    alpaka::memcpy(queue,
                   alpaka::createView(queue, d_points.view().m_coords[Dim], h_points.size()),
                   alpaka::createView(host, h_points.view().m_coords[Dim], h_points.size()));
  });
  alpaka::memcpy(queue,
                 alpaka::createView(queue, d_points.view().m_weight, h_points.size()),
                 alpaka::createView(host, h_points.view().m_weight, h_points.size()));
  if (h_points.view().has_uncertainty()) {
    using dev_value_t = std::remove_cv_t<TDeviceInput>;
    using PType = std::remove_cvref_t<decltype(d_points)>;
    auto& ubuf = internal::points_interface<PType>::uncertainty_buffer(d_points);
    ubuf = internal::alloc_device_buffer<dev_value_t>(
        queue, static_cast<std::size_t>(h_points.size()), allocator);
    alpaka::memcpy(
        queue,
        alpaka::createView(queue, ubuf->data(), h_points.size()),
        alpaka::createView(host, h_points.view().m_density_uncertainty, h_points.size()));
    d_points.view().m_density_uncertainty = ubuf->data();
  }
  meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
    if (h_points.view().has_sigma(Dim)) {
      using dev_value_t = std::remove_cv_t<TDeviceInput>;
      using PType = std::remove_cvref_t<decltype(d_points)>;
      auto& cbufs = internal::points_interface<PType>::sigma_buffers(d_points);
      cbufs[Dim] = internal::alloc_device_buffer<dev_value_t>(
          queue, static_cast<std::size_t>(h_points.size()), allocator);
      alpaka::memcpy(queue,
                     alpaka::createView(queue, cbufs[Dim]->data(), h_points.size()),
                     alpaka::createView(host, h_points.view().m_sigmas[Dim], h_points.size()));
      d_points.view().m_sigmas[Dim] = cbufs[Dim]->data();
    }
  });
  if (h_points.view().has_tags()) {
    using PType = std::remove_cvref_t<decltype(d_points)>;
    auto& tbuf = internal::points_interface<PType>::tags_buffer(d_points);
    tbuf = internal::alloc_device_buffer<std::size_t>(
        queue, static_cast<std::size_t>(h_points.size()), allocator);
    alpaka::memcpy(queue,
                   alpaka::createView(queue, tbuf->data(), h_points.size()),
                   alpaka::createView(host, h_points.view().m_tags, h_points.size()));
    d_points.view().m_tags = tbuf->data();
  }
  internal::setup_details_buffers(queue, d_points, h_points, allocator);
}

CLUE_NAMESPACE_END
