
#pragma once

#include <cstddef>
#include <cstdint>
#include <concepts>
#include <span>
#include <stdexcept>
#include <tuple>

namespace clue::soa::device {

  template <std::size_t Ndim, std::floating_point TValue>
  inline auto computeSoASize(std::int32_t n_points) {
    if (n_points <= 0) {
      throw std::invalid_argument(
          "Number of points passed to PointsDevice constructor must be positive.");
    }
    return ((Ndim + 2) * sizeof(TValue) + 3 * sizeof(int)) * n_points;
  }

  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* buffer,
                               std::int32_t n_points) {
    using value_type = std::remove_cv_t<TElement>;

    meta::apply<Ndim>([&]<std::size_t Dim>() {
      view.m_coords[Dim] =
          reinterpret_cast<value_type*>(buffer + Dim * n_points * sizeof(value_type));
    });
    view.m_weight = reinterpret_cast<value_type*>(buffer + Ndim * n_points * sizeof(value_type));
    view.m_cluster_index =
        reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(value_type));
    view.m_is_seed =
        reinterpret_cast<int*>(buffer + n_points * ((Ndim + 1) * sizeof(value_type) + sizeof(int)));
    view.m_rho = reinterpret_cast<value_type*>(
        buffer + n_points * ((Ndim + 1) * sizeof(value_type) + 2 * sizeof(int)));
    view.m_nearest_higher = reinterpret_cast<int*>(
        buffer + n_points * ((Ndim + 2) * sizeof(value_type) + 2 * sizeof(int)));
    view.m_n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* alloc_buffer,
                               std::byte* buffer,
                               std::int32_t n_points) {
    using value_type = std::remove_cv_t<TElement>;

    meta::apply<Ndim>([&]<std::size_t Dim>() {
      view.m_coords[Dim] =
          reinterpret_cast<value_type*>(buffer + Dim * n_points * sizeof(value_type));
    });
    view.m_weight = reinterpret_cast<value_type*>(buffer + Ndim * n_points * sizeof(value_type));
    view.m_cluster_index =
        reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(value_type));
    view.m_is_seed = reinterpret_cast<int*>(alloc_buffer);
    view.m_rho = reinterpret_cast<value_type*>(alloc_buffer + n_points * sizeof(int));
    view.m_nearest_higher =
        reinterpret_cast<int*>(alloc_buffer + n_points * (sizeof(value_type) + sizeof(int)));
    view.m_n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* alloc_buffer,
                               std::int32_t n_points,
                               std::span<TElement> coordinates,
                               std::span<TElement> weights,
                               std::span<int> output) {
    using value_type = std::remove_cv_t<TElement>;

    meta::apply<Ndim>(
        [&]<std::size_t Dim>() { view.m_coords[Dim] = coordinates.data() + Dim * n_points; });
    view.m_weight = weights.data();
    view.m_cluster_index = output.data();
    view.m_is_seed = reinterpret_cast<int*>(alloc_buffer);
    view.m_rho = reinterpret_cast<value_type*>(alloc_buffer + n_points * sizeof(value_type));
    view.m_nearest_higher =
        reinterpret_cast<int*>(alloc_buffer + n_points * (sizeof(value_type) + sizeof(int)));
    view.m_n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* alloc_buffer,
                               std::int32_t n_points,
                               std::span<TElement> input,
                               std::span<int> output) {
    using value_type = std::remove_cv_t<TElement>;

    meta::apply<Ndim>(
        [&]<std::size_t Dim>() { view.m_coords[Dim] = input.data() + Dim * n_points; });
    view.m_weight = input.data() + Ndim * n_points;
    view.m_cluster_index = output.data();
    view.m_is_seed = reinterpret_cast<int*>(alloc_buffer);
    view.m_rho = reinterpret_cast<value_type*>(alloc_buffer + n_points * sizeof(value_type));
    view.m_nearest_higher =
        reinterpret_cast<int*>(alloc_buffer + n_points * (sizeof(value_type) + sizeof(int)));
    view.m_n = n_points;
  }

  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* alloc_buffer,
                               std::int32_t n_points,
                               TElement* coordinates,
                               TElement* weights,
                               int* output) {
    using value_type = std::remove_cv_t<TElement>;

    meta::apply<Ndim>(
        [&]<std::size_t Dim>() { view.m_coords[Dim] = coordinates + Dim * n_points; });
    view.m_weight = weights;
    view.m_cluster_index = output;
    view.m_is_seed = reinterpret_cast<int*>(alloc_buffer);
    view.m_rho = reinterpret_cast<value_type*>(alloc_buffer + n_points * sizeof(value_type));
    view.m_nearest_higher =
        reinterpret_cast<int*>(alloc_buffer + n_points * (sizeof(value_type) + sizeof(int)));
    view.m_n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* alloc_buffer,
                               std::int32_t n_points,
                               TElement* input,
                               int* output) {
    using value_type = std::remove_cv_t<TElement>;

    meta::apply<Ndim>([&]<std::size_t Dim>() { view.m_coords[Dim] = input + Dim * n_points; });
    view.m_weight = input + Ndim * n_points;
    view.m_cluster_index = output;
    view.m_is_seed = reinterpret_cast<int*>(alloc_buffer);
    view.m_rho = reinterpret_cast<value_type*>(alloc_buffer + n_points * sizeof(value_type));
    view.m_nearest_higher =
        reinterpret_cast<int*>(alloc_buffer + n_points * (sizeof(value_type) + sizeof(int)));
    view.m_n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* alloc_buffer,
                               std::int32_t n_points,
                               TBuffers... buffers) {
    using value_type = std::remove_cv_t<TElement>;
    auto buffers_tuple = std::make_tuple(buffers...);

    meta::apply<Ndim>(
        [&]<std::size_t Dim>() { view.m_coords[Dim] = std::get<Dim>(buffers_tuple); });
    view.m_weight = std::get<Ndim>(buffers_tuple);
    view.m_cluster_index = std::get<Ndim + 1>(buffers_tuple);
    view.m_is_seed = reinterpret_cast<int*>(alloc_buffer);
    view.m_rho = reinterpret_cast<value_type*>(alloc_buffer + n_points * sizeof(value_type));
    view.m_nearest_higher =
        reinterpret_cast<int*>(alloc_buffer + n_points * (sizeof(value_type) + sizeof(int)));
    view.m_n = n_points;
  }

}  // namespace clue::soa::device
