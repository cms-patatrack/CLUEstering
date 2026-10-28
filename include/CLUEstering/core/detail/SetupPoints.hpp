
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/data_structures/PointsConversion.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/default_allocator.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

CLUE_NAMESPACE_BEGIN

namespace detail {

  // On CPU the device points alias the host memory, so nothing is allocated for the coordinates
  // and the allocator is not used
  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TInput,
            concepts::allocator TAllocator = DefaultAllocator>
    requires std::same_as<alpaka::Dev<TQueue>, alpaka::DevCpu>
  inline auto make_device_points(TQueue& queue,
                                 PointsHost<Ndim, TInput>& points,
                                 const TAllocator& /* allocator */ = TAllocator{}) {
    using PointsDev = PointsDevice<Ndim, TInput, alpaka::DevCpu>;

    auto& view = points.view();
    auto dev_points = [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> PointsDev {
      return PointsDev(queue,
                       points.size(),
                       view.coords()[Dims].data()...,
                       view.weights().data(),
                       view.cluster_index().data());
    }(std::make_index_sequence<Ndim>{});
    dev_points.view().m_sigmas = view.m_sigmas;
    dev_points.view().m_density_uncertainty = view.m_density_uncertainty;
    dev_points.view().m_tags = view.m_tags;

    return dev_points;
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TInput,
            concepts::allocator TAllocator = DefaultAllocator>
    requires(!std::same_as<alpaka::Dev<TQueue>, alpaka::DevCpu>)
  inline auto make_device_points(TQueue& queue,
                                 PointsHost<Ndim, TInput>& points,
                                 const TAllocator& allocator = TAllocator{}) {
    using value_type = std::remove_cv_t<TInput>;
    using PointsDev = PointsDevice<Ndim, value_type, alpaka::Dev<TQueue>>;

    PointsDev dev_points(queue, points.size(), allocator);
    copyToDevice(queue, dev_points, points, allocator);

    return dev_points;
  }

  template <concepts::queue TQueue, typename TPointsHost, typename TPointsDevice>
    requires std::same_as<alpaka::Dev<TQueue>, alpaka::DevCpu>
  inline void sync_results(TQueue& /* queue */,
                           TPointsHost& /* h_points */,
                           const TPointsDevice& /* d_points */) { /* no-op for CPU */ }

  template <concepts::queue TQueue, typename TPointsHost, typename TPointsDevice>
    requires(!std::same_as<alpaka::Dev<TQueue>, alpaka::DevCpu>)
  inline void sync_results(TQueue& queue, TPointsHost& h_points, const TPointsDevice& d_points) {
    copyToHost(queue, h_points, d_points);
  }

}  // namespace detail

CLUE_NAMESPACE_END
