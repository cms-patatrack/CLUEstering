
#pragma once

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/ClusterProperties.hpp"
#include "CLUEstering/utils/detail/get_cluster_properties.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/meta/apply.hpp"

#include <alpaka/alpaka.hpp>
#include <cassert>
#include <concepts>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>

namespace clue {

  namespace soa::host {

    // No need to allocate temporary buffers on the host
    template <std::size_t Ndim, std::floating_point TData>
    inline auto computeSoASize(std::int32_t n_points) {
      if (n_points <= 0) {
        throw std::invalid_argument(
            "Number of points passed to PointsHost constructor must be positive.");
      }
      return ((Ndim + 1) * sizeof(TData) + sizeof(int)) * n_points;
    }

    template <std::size_t Ndim, std::floating_point TData>
    inline void partitionSoAView(PointsView<Ndim, TData>& view,
                                 std::byte* buffer,
                                 int32_t n_points) {
      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<TData*>(buffer + Dim * n_points * sizeof(TData));
      });
      view.weight = reinterpret_cast<TData*>(buffer + Ndim * n_points * sizeof(TData));
      view.cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(TData));
      view.n = n_points;
    }
    template <std::size_t Ndim, std::floating_point TData, concepts::pointer... TBuffers>
      requires(sizeof...(TBuffers) == 3)
    inline void partitionSoAView(PointsView<Ndim, TData>& view,
                                 int32_t n_points,
                                 TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<TData*>(std::get<0>(buffers_tuple) + Dim * n_points);
      });
      view.weight = std::get<1>(buffers_tuple);
      view.cluster_index = std::get<2>(buffers_tuple);
      view.n = n_points;
    }
    template <std::size_t Ndim, std::floating_point TData, concepts::pointer... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    inline void partitionSoAView(PointsView<Ndim, TData>& view,
                                 int32_t n_points,
                                 TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<TData*>(std::get<0>(buffers_tuple) + Dim * n_points);
      });
      view.weight = std::get<0>(buffers_tuple) + Ndim * n_points;
      view.cluster_index = std::get<1>(buffers_tuple);
      view.n = n_points;
    }
    template <std::size_t Ndim, std::floating_point TData, concepts::pointer... TBuffers>
      requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
    inline void partitionSoAView(PointsView<Ndim, TData>& view,
                                 int32_t n_points,
                                 TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      meta::apply<Ndim>(
          [&]<std::size_t Dim>() { view.coords[Dim] = std::get<Dim>(buffers_tuple); });
      view.weight = std::get<Ndim>(buffers_tuple) + Ndim * n_points;
      view.cluster_index = std::get<Ndim + 1>(buffers_tuple);
      view.n = n_points;
    }

    template <std::size_t Ndim, std::floating_point TData, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 3)
    inline void partitionSoAView(PointsView<Ndim, TData>& view,
                                 int32_t n_points,
                                 TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] =
            reinterpret_cast<TData*>(std::get<0>(buffers_tuple).data() + Dim * n_points);
      });
      view.weight = std::get<1>(buffers_tuple).data();
      view.cluster_index = std::get<2>(buffers_tuple).data();
      view.n = n_points;
    }
    template <std::size_t Ndim, std::floating_point TData, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    inline void partitionSoAView(PointsView<Ndim, TData>& view,
                                 int32_t n_points,
                                 TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] =
            reinterpret_cast<TData*>(std::get<0>(buffers_tuple).data() + Dim * n_points);
      });
      view.weight = std::get<0>(buffers_tuple).data() + Ndim * n_points;
      view.cluster_index = std::get<1>(buffers_tuple).data();
      view.n = n_points;
    }
    template <std::size_t Ndim, std::floating_point TData, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
    inline void partitionSoAView(PointsView<Ndim, TData>& view,
                                 int32_t n_points,
                                 TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] =
            reinterpret_cast<TData*>(std::get<0>(buffers_tuple).data() + Dim * n_points);
      });
      view.weight = std::get<0>(buffers_tuple).data() + Ndim * n_points;
      view.cluster_index = std::get<1>(buffers_tuple).data();
      view.n = n_points;
    }

  }  // namespace soa::host

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue& queue, int32_t n_points)
      : m_buffer{make_host_buffer<std::byte[]>(
            queue, soa::host::computeSoASize<Ndim, value_type>(n_points))},
        m_view{},
        m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, m_buffer->data(), n_points);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&, int32_t n_points, std::span<std::byte> buffer)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, buffer.data(), n_points);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&,
                                             int32_t n_points,
                                             std::span<value_type> input,
                                             std::span<int> output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, input, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&,
                                             int32_t n_points,
                                             std::span<value_type> coordinates,
                                             std::span<value_type> weights,
                                             std::span<int> output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&, int32_t n_points, TBuffers&&... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, std::forward<TBuffers>(buffers)...);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&,
                                             int32_t n_points,
                                             value_type* input,
                                             int* output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, input, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(
      TQueue&, int32_t n_points, value_type* coordinates, value_type* weights, int* output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&, int32_t n_points, TBuffers... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, buffers...);
  }

  template <std::size_t Ndim, std::floating_point TData>
  inline PointsHost<Ndim, TData>::Point PointsHost<Ndim, TData>::operator[](std::size_t idx) const {
    if (idx >= static_cast<size_t>(m_size))
      throw std::out_of_range("Index out of range in PointsHost::operator[]");

    std::array<TData, Ndim> coords;
    for (size_t dim = 0; dim < Ndim; ++dim) {
      coords[dim] = m_view.coords[0][dim * m_size + idx];
    }
    return Point(coords, m_view.weight[idx], m_view.cluster_index[idx]);
  }

  template <std::size_t Ndim, std::floating_point TData>
  inline const auto& PointsHost<Ndim, TData>::n_clusters() {
    assert(m_clustered &&
           "The points have to be clustered before the cluster properties can be accessed");
    if (m_clusterProperties.has_value())
      return m_clusterProperties->n_clusters();
    if (!m_nclusters.has_value())
      m_nclusters = detail::compute_nclusters(this->clusterIndexes());

    return m_nclusters.value();
  }

  template <std::size_t Ndim, std::floating_point TData>
  inline const auto& PointsHost<Ndim, TData>::clusters() {
    assert(m_clustered &&
           "The points have to be clustered before the cluster properties can be accessed");
    if (!m_clusterProperties.has_value())
      m_clusterProperties = ClusterProperties{this->clusterIndexes()};

    return m_clusterProperties->m_clusters_to_points;
  }

  template <std::size_t Ndim, std::floating_point TData>
  inline const auto& PointsHost<Ndim, TData>::cluster_sizes() {
    assert(m_clustered &&
           "The points have to be clustered before the cluster properties can be accessed");
    if (!m_clusterProperties.has_value())
      m_clusterProperties = ClusterProperties{this->clusterIndexes()};

    return m_clusterProperties->m_cluster_sizes;
  }

  template <std::size_t Ndim, std::floating_point TData>
  inline const auto& PointsHost<Ndim, TData>::cluster_properties() {
    assert(m_clustered &&
           "The points have to be clustered before the cluster properties can be accessed");
    if (!m_clusterProperties.has_value())
      m_clusterProperties = ClusterProperties{this->clusterIndexes()};

    return m_clusterProperties.value();
  }

}  // namespace clue
