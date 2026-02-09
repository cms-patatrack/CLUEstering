
#pragma once

#include <cstddef>
#include <cstdint>
#include <concepts>
#include <span>
#include <stdexcept>
#include <tuple>

namespace clue::soa::host {

  // No need to allocate temporary buffers on the host
  template <std::size_t Ndim, std::floating_point TValue>
  inline auto computeSoASize(std::int32_t n_points) {
    if (n_points <= 0) {
      throw std::invalid_argument(
          "Number of points passed to PointsHost constructor must be positive.");
    }
    return ((Ndim + 1) * sizeof(TValue) + sizeof(int)) * n_points;
  }

  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               std::byte* buffer,
                               int32_t n_points) {
    using value_type = std::remove_cv_t<TElement>;

    meta::apply<Ndim>([&]<std::size_t Dim>() {
      view.coords[Dim] =
          reinterpret_cast<value_type*>(buffer + Dim * n_points * sizeof(value_type));
    });
    view.weight = reinterpret_cast<value_type*>(buffer + Ndim * n_points * sizeof(value_type));
    view.cluster_index =
        reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(value_type));
    view.n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               int32_t n_points,
                               TElement* coordinates,
                               TElement* weights,
                               int* cluster_indices) {
    meta::apply<Ndim>([&]<std::size_t Dim>() { view.coords[Dim] = coordinates + Dim * n_points; });
    view.weight = weights;
    view.cluster_index = cluster_indices;
    view.n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               int32_t n_points,
                               TElement* input,
                               int* output) {
    meta::apply<Ndim>([&]<std::size_t Dim>() { view.coords[Dim] = input + Dim * n_points; });
    view.weight = input + Ndim * n_points;
    view.cluster_index = output;
    view.n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               int32_t n_points,
                               TBuffers... buffers) {
    auto buffers_tuple = std::make_tuple(buffers...);

    meta::apply<Ndim>([&]<std::size_t Dim>() { view.coords[Dim] = std::get<Dim>(buffers_tuple); });
    view.weight = std::get<Ndim>(buffers_tuple) + Ndim * n_points;
    view.cluster_index = std::get<Ndim + 1>(buffers_tuple);
    view.n = n_points;
  }

  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               int32_t n_points,
                               std::span<TElement> coordinates,
                               std::span<TElement> weights,
                               std::span<int> cluster_indices) {
    meta::apply<Ndim>(
        [&]<std::size_t Dim>() { view.coords[Dim] = coordinates.data() + Dim * n_points; });
    view.weight = weights.data();
    view.cluster_index = cluster_indices.data();
    view.n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement>
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               int32_t n_points,
                               std::span<TElement> input,
                               std::span<int> cluster_indices) {
    meta::apply<Ndim>([&]<std::size_t Dim>() { view.coords[Dim] = input.data() + Dim * n_points; });
    view.weight = input.data() + Ndim * n_points;
    view.cluster_index = cluster_indices.data();
    view.n = n_points;
  }
  template <std::size_t Ndim, std::floating_point TElement, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline void partitionSoAView(PointsView<Ndim, TElement>& view,
                               int32_t n_points,
                               TBuffers&&... buffers) {
    auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);

    meta::apply<Ndim>([&]<std::size_t Dim>() {
      view.coords[Dim] = std::get<0>(buffers_tuple).data() + Dim * n_points;
    });
    view.weight = std::get<0>(buffers_tuple).data() + Ndim * n_points;
    view.cluster_index = std::get<1>(buffers_tuple).data();
    view.n = n_points;
  }

}  // namespace clue::soa::host
