
#pragma once

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/ClusterProperties.hpp"
#include "CLUEstering/utils/detail/get_cluster_properties.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>

namespace clue {

  namespace soa::host {

    // No need to allocate temporary buffers on the host
    template <std::size_t Ndim>
    inline auto computeSoASize(int32_t n_points) {
      if (n_points <= 0) {
        throw std::invalid_argument(
            "Number of points passed to PointsHost constructor must be positive.");
      }
      return ((Ndim + 1) * sizeof(float) + sizeof(int)) * n_points;
    }

    template <std::size_t Ndim>
    inline void partitionSoAView(PointsView<Ndim>& view, std::byte* buffer, int32_t n_points) {
      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((view.coords[Dims] = reinterpret_cast<float*>(buffer + Dims * n_points * sizeof(float))),
         ...);
      }(std::make_index_sequence<Ndim>{});
      view.weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view.cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view.n = n_points;
    }
    template <std::size_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 3)
    inline void partitionSoAView(PointsView<Ndim>& view, int32_t n_points, TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);

      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((view.coords[Dims] =
              reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Dims * n_points)),
         ...);
      }(std::make_index_sequence<Ndim>{});
      view.weight = std::get<1>(buffers_tuple);
      view.cluster_index = std::get<2>(buffers_tuple);
      view.n = n_points;
    }
    template <std::size_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    inline void partitionSoAView(PointsView<Ndim>& view, int32_t n_points, TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((view.coords[Dims] =
              reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Dims * n_points)),
         ...);
      }(std::make_index_sequence<Ndim>{});
      view.weight = std::get<0>(buffers_tuple) + Ndim * n_points;
      view.cluster_index = std::get<1>(buffers_tuple);
      view.n = n_points;
    }
    template <std::size_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
    inline void partitionSoAView(PointsView<Ndim>& view, int32_t n_points, TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((view.coords[Dims] = std::get<Dims>(buffers_tuple)), ...);
      }(std::make_index_sequence<Ndim>{});
      view.weight = std::get<Ndim>(buffers_tuple) + Ndim * n_points;
      view.cluster_index = std::get<Ndim + 1>(buffers_tuple);
      view.n = n_points;
    }

    template <std::size_t Ndim, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 3)
    inline void partitionSoAView(PointsView<Ndim>& view, int32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);

      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((view.coords[Dims] =
              reinterpret_cast<float*>(std::get<0>(buffers_tuple).data() + Dims * n_points)),
         ...);
      }(std::make_index_sequence<Ndim>{});
      view.weight = std::get<1>(buffers_tuple).data();
      view.cluster_index = std::get<2>(buffers_tuple).data();
      view.n = n_points;
    }
    template <std::size_t Ndim, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    inline void partitionSoAView(PointsView<Ndim>& view, int32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);

      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((view.coords[Dims] =
              reinterpret_cast<float*>(std::get<0>(buffers_tuple).data() + Dims * n_points)),
         ...);
      }(std::make_index_sequence<Ndim>{});
      view.weight = std::get<0>(buffers_tuple).data() + Ndim * n_points;
      view.cluster_index = std::get<1>(buffers_tuple).data();
      view.n = n_points;
    }
    template <uint8_t Ndim, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
    inline void partitionSoAView(PointsView<Ndim>& view, int32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);

      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((view.coords[Dims] =
              reinterpret_cast<float*>(std::get<0>(buffers_tuple).data() + Dims * n_points)),
         ...);
      }(std::make_index_sequence<Ndim>{});
      view.weight = std::get<0>(buffers_tuple).data() + Ndim * n_points;
      view.cluster_index = std::get<1>(buffers_tuple).data();
      view.n = n_points;
    }

  }  // namespace soa::host

  template <std::size_t Ndim>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points)
      : m_buffer{make_host_buffer<std::byte[]>(queue, soa::host::computeSoASize<Ndim>(n_points))},
        m_view{},
        m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, m_buffer->data(), n_points);
  }

  template <std::size_t Ndim>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points, std::span<std::byte> buffer)
      : m_view{}, m_size{n_points} {
    assert(buffer.size() == soa::host::computeSoASize<Ndim>(n_points));

    soa::host::partitionSoAView<Ndim>(m_view, buffer.data(), n_points);
  }

  template <std::size_t Ndim>
  template <concepts::queue TQueue, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 3 ||
             (sizeof...(TBuffers) == Ndim + 2 and Ndim > 1))
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points, TBuffers&&... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, std::forward<TBuffers>(buffers)...);
  }

  template <std::size_t Ndim>
  template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
    requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 3 ||
             (sizeof...(TBuffers) == Ndim + 2 and Ndim > 1))
  inline PointsHost<Ndim>::PointsHost(TQueue& queue, int32_t n_points, TBuffers... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView<Ndim>(m_view, n_points, buffers...);
  }

  template <std::size_t Ndim>
  inline PointsHost<Ndim>::Point PointsHost<Ndim>::operator[](std::size_t idx) const {
    if (idx >= static_cast<size_t>(m_size))
      throw std::out_of_range("Index out of range in PointsHost::operator[]");

    std::array<float, Ndim> coords;
    for (size_t dim = 0; dim < Ndim; ++dim) {
      coords[dim] = m_view.coords[0][dim * m_size + idx];
    }
    return Point(coords, m_view.weight[idx], m_view.cluster_index[idx]);
  }

  template <std::size_t Ndim>
  inline const auto& PointsHost<Ndim>::n_clusters() {
    assert(("The points have to be clustered before the cluster properties can be accessed",
            m_clustered));
    if (m_clusterProperties.has_value())
      return m_clusterProperties->n_clusters();
    if (!m_nclusters.has_value())
      m_nclusters = detail::compute_nclusters(this->clusterIndexes());

    return m_nclusters.value();
  }

  template <std::size_t Ndim>
  inline const auto& PointsHost<Ndim>::clusters() {
    assert(("The points have to be clustered before the cluster properties can be accessed",
            m_clustered));
    if (!m_clusterProperties.has_value())
      m_clusterProperties = ClusterProperties{this->clusterIndexes()};

    return m_clusterProperties->m_clusters_to_points;
  }

  template <std::size_t Ndim>
  inline const auto& PointsHost<Ndim>::cluster_sizes() {
    assert(("The points have to be clustered before the cluster properties can be accessed",
            m_clustered));
    if (!m_clusterProperties.has_value())
      m_clusterProperties = ClusterProperties{this->clusterIndexes()};

    return m_clusterProperties->m_cluster_sizes;
  }

  template <std::size_t Ndim>
  inline const auto& PointsHost<Ndim>::cluster_properties() {
    assert(("The points have to be clustered before the cluster properties can be accessed",
            m_clustered));
    if (!m_clusterProperties.has_value())
      m_clusterProperties = ClusterProperties{this->clusterIndexes()};

    return m_clusterProperties.value();
  }

}  // namespace clue
