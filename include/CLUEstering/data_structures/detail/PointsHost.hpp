
#pragma once

#include "CLUEstering/data_structures/ClusterProperties.hpp"
#include "CLUEstering/data_structures/detail/HostViewPartition.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
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

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue& queue, int32_t n_points)
      : m_buffer{make_host_buffer<std::byte[]>(
            queue, soa::host::computeSoASize<Ndim, value_type>(n_points))},
        m_view{},
        m_size{n_points} {
    soa::host::partitionSoAView(m_view, m_buffer->data(), n_points);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&, int32_t n_points, std::span<std::byte> buffer)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView(m_view, buffer.data(), n_points);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&,
                                             int32_t n_points,
                                             std::span<element_type> input,
                                             std::span<int> output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView(m_view, n_points, input, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&,
                                             int32_t n_points,
                                             std::span<element_type> coordinates,
                                             std::span<element_type> weights,
                                             std::span<int> output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView(m_view, n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&, int32_t n_points, TBuffers&&... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView(m_view, n_points, std::forward<TBuffers>(buffers)...);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&,
                                             int32_t n_points,
                                             element_type* input,
                                             int* output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView(m_view, n_points, input, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue>
  inline PointsHost<Ndim, TData>::PointsHost(
      TQueue&, int32_t n_points, element_type* coordinates, element_type* weights, int* output)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView(m_view, n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, std::floating_point TData>
  template <concepts::queue TQueue, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline PointsHost<Ndim, TData>::PointsHost(TQueue&, int32_t n_points, TBuffers... buffers)
      : m_view{}, m_size{n_points} {
    soa::host::partitionSoAView(m_view, n_points, buffers...);
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
