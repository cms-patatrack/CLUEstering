
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/PointsFactory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/utils/get_queue.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace clue {

  namespace detail {

    template <std::size_t Ndim, std::floating_point TData, typename TContainer>
    struct points_type {};

    template <std::size_t Ndim, std::floating_point TData, concepts::queue TQueue>
    struct points_type<Ndim, TData, TQueue> {
      using device_type = decltype(alpaka::getDev(std::declval<TQueue>()));
      using type = std::conditional_t<std::is_same_v<device_type, alpaka::DevCpu>,
                                      PointsHost<Ndim, const TData>,
                                      PointsDevice<Ndim, const TData, device_type>>;
    };

    template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
    struct points_type<Ndim, TData, TDev> {
      using type = std::conditional_t<std::is_same_v<TDev, alpaka::DevCpu>,
                                      PointsHost<Ndim, const TData>,
                                      PointsDevice<Ndim, const TData, TDev>>;
    };

  }  // namespace detail

  template <std::size_t Ndim, std::floating_point InputType, concepts::queue QueueType>
  auto make_clustered_points(QueueType& queue,
                             std::size_t n_points,
                             const InputType* coordinates,
                             const InputType* weights,
                             const int* cluster_indexes) {
    using points_type = typename detail::points_type<Ndim, InputType, QueueType>::type;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto points = points_type(queue,
                              static_cast<int32_t>(n_points),
                              coordinates,
                              weights,
                              const_cast<int*>(cluster_indexes));
    internal::points_interface<points_type>::mark_clustered(points);
    return points;
  }

  template <std::size_t Ndim, std::floating_point InputType, concepts::queue QueueType>
  auto make_clustered_points(QueueType& queue,
                             std::span<const InputType> coordinates,
                             std::span<const InputType> weights,
                             std::span<const int> cluster_indexes) {
    using points_type = typename detail::points_type<Ndim, InputType, QueueType>::type;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto ci = std::span<int>(const_cast<int*>(cluster_indexes.data()), cluster_indexes.size());
    auto points =
        points_type(queue, static_cast<int32_t>(cluster_indexes.size()), coordinates, weights, ci);
    internal::points_interface<points_type>::mark_clustered(points);
    return points;
  }

  template <std::size_t Ndim, std::floating_point InputType, concepts::device DeviceType>
  auto make_clustered_points(const DeviceType& device,
                             std::size_t n_points,
                             const InputType* coordinates,
                             const InputType* weights,
                             const int* cluster_indexes) {
    auto queue = clue::get_queue(device);
    return make_clustered_points<Ndim>(queue, n_points, coordinates, weights, cluster_indexes);
  }

  template <std::size_t Ndim, std::floating_point InputType, concepts::device DeviceType>
  auto make_clustered_points(const DeviceType& device,
                             std::span<const InputType> coordinates,
                             std::span<const InputType> weights,
                             std::span<const int> cluster_indexes) {
    auto queue = clue::get_queue(device);
    return make_clustered_points<Ndim>(queue, coordinates, weights, cluster_indexes);
  }

  template <std::size_t Ndim, concepts::queue QueueType, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(QueueType& queue, std::size_t n_points, TBuffers... buffers) {
    using first_buffer_t = std::tuple_element_t<0, std::tuple<TBuffers...>>;
    using InputType = std::remove_cv_t<std::remove_pointer_t<first_buffer_t>>;
    static_assert(std::floating_point<InputType>,
                  "The first Ndim buffers must be floating-point coordinate arrays");
    using PType = typename detail::points_type<Ndim, InputType, QueueType>::type;
    auto buffers_tuple = std::make_tuple(buffers...);
    auto points = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> PType {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      auto* cluster_idx = const_cast<int*>(std::get<Ndim + 1>(buffers_tuple));
      return PType(
          queue, static_cast<int32_t>(n_points), std::get<Is>(buffers_tuple)..., cluster_idx);
    }(std::make_index_sequence<Ndim + 1>{});
    internal::points_interface<PType>::mark_clustered(points);
    return points;
  }

  template <std::size_t Ndim, concepts::queue QueueType, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(QueueType& queue, TBuffers&&... buffers) {
    using first_buffer_t = std::remove_cvref_t<std::tuple_element_t<0, std::tuple<TBuffers...>>>;
    using InputType = std::remove_cv_t<std::ranges::range_value_t<first_buffer_t>>;
    static_assert(std::floating_point<InputType>,
                  "The first Ndim buffers must be floating-point coordinate arrays");
    using PType = typename detail::points_type<Ndim, InputType, QueueType>::type;
    auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);
    const auto n_points = static_cast<int32_t>(std::get<0>(buffers_tuple).size());
    auto points = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> PType {
      auto& ci_span = std::get<Ndim + 1>(buffers_tuple);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      auto* cluster_idx = const_cast<int*>(ci_span.data());
      return PType(queue, n_points, std::get<Is>(buffers_tuple).data()..., cluster_idx);
    }(std::make_index_sequence<Ndim + 1>{});
    internal::points_interface<PType>::mark_clustered(points);
    return points;
  }

  template <std::size_t Ndim, concepts::device DeviceType, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(const DeviceType& device, std::size_t n_points, TBuffers... buffers) {
    auto queue = clue::get_queue(device);
    return make_clustered_points<Ndim>(queue, n_points, buffers...);
  }

  template <std::size_t Ndim, concepts::device DeviceType, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(const DeviceType& device, TBuffers&&... buffers) {
    auto queue = clue::get_queue(device);
    return make_clustered_points<Ndim>(queue, std::forward<TBuffers>(buffers)...);
  }

}  // namespace clue
