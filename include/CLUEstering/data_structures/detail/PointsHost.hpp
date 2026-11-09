
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/data_structures/ClusterProperties.hpp"
#include "CLUEstering/data_structures/detail/HostViewPartition.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/utils/detail/get_cluster_properties.hpp"
#include "CLUEstering/internal/alpaka/buffer.hpp"
#include "CLUEstering/internal/meta/apply.hpp"

#include <alpaka/alpaka.hpp>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>

CLUE_NAMESPACE_BEGIN

template <std::size_t Ndim, std::floating_point TData>
template <concepts::queue TQueue>
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */, int32_t n_points)
    : m_buffer{alpaka::allocMappedBuf<std::byte>(
          host, platform<TQueue>(), soa::host::computeSoASize<Ndim, value_type>(n_points))},
      m_view{},
      m_size{n_points} {
  soa::host::partitionSoAView(m_view, m_buffer->data(), n_points);
}

template <std::size_t Ndim, std::floating_point TData>
template <concepts::queue TQueue>
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */,
                                           int32_t n_points,
                                           std::span<std::byte> buffer)
    : m_view{}, m_size{n_points} {
  soa::host::partitionSoAView(m_view, buffer.data(), n_points);
}

template <std::size_t Ndim, std::floating_point TData>
template <concepts::queue TQueue>
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */,
                                           int32_t n_points,
                                           std::span<element_type> input,
                                           std::span<int> output)
    : m_view{}, m_size{n_points} {
  soa::host::partitionSoAView(m_view, n_points, input, output);
}

template <std::size_t Ndim, std::floating_point TData>
template <concepts::queue TQueue>
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */,
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
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */,
                                           int32_t n_points,
                                           TBuffers&&... buffers)
    : m_view{}, m_size{n_points} {
  soa::host::partitionSoAView(m_view, n_points, std::forward<TBuffers>(buffers)...);
}

template <std::size_t Ndim, std::floating_point TData>
template <concepts::queue TQueue>
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */,
                                           int32_t n_points,
                                           element_type* input,
                                           int* output)
    : m_view{}, m_size{n_points} {
  soa::host::partitionSoAView(m_view, n_points, input, output);
}

template <std::size_t Ndim, std::floating_point TData>
template <concepts::queue TQueue>
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */,
                                           int32_t n_points,
                                           element_type* coordinates,
                                           element_type* weights,
                                           int* output)
    : m_view{}, m_size{n_points} {
  soa::host::partitionSoAView(m_view, n_points, coordinates, weights, output);
}

template <std::size_t Ndim, std::floating_point TData>
template <concepts::queue TQueue, concepts::pointer... TBuffers>
  requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
inline PointsHost<Ndim, TData>::PointsHost(TQueue& /* queue */,
                                           int32_t n_points,
                                           TBuffers... buffers)
    : m_view{}, m_size{n_points} {
  soa::host::partitionSoAView(m_view, n_points, buffers...);
}

template <std::size_t Ndim, std::floating_point TData>
inline PointsHost<Ndim, TData>::Point PointsHost<Ndim, TData>::operator[](std::size_t idx) const {
  if (idx >= static_cast<size_t>(m_size))
    throw std::out_of_range("Index out of range in PointsHost::operator[]");

  std::array<TData, Ndim> coords;
  for (auto dim = 0u; dim < Ndim; ++dim) {
    coords[dim] = m_view.coords()[dim][idx];
  }
  return Point(coords, m_view.weights()[idx], m_view.cluster_index()[idx]);
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

template <std::size_t Ndim, std::floating_point TData>
inline void PointsHost<Ndim, TData>::store_details() {
  if (has_details())
    return;

  const auto n_points = static_cast<std::size_t>(m_size);
  m_density = alpaka::allocMappedBuf<value_type>(host, platform<Platform>(), n_points);
  m_delta = alpaka::allocMappedBuf<value_type>(host, platform<Platform>(), n_points);
  m_nearest_higher = alpaka::allocMappedBuf<std::int32_t>(host, platform<Platform>(), n_points);
  m_view.m_rho = m_density->data();
  m_view.m_delta = m_delta->data();
  m_view.m_nearest_higher = m_nearest_higher->data();
}

template <std::size_t Ndim, std::floating_point TData>
inline bool PointsHost<Ndim, TData>::has_details() const {
  return m_density.has_value();
}

template <std::size_t Ndim, std::floating_point TData>
inline auto PointsHost<Ndim, TData>::rho() const {
  assert(m_clustered && "The points have to be clustered before the densities can be accessed");
  assert(has_details() && "store_details must be called before clustering to access rho");
  return std::as_const(m_view).rho();
}

template <std::size_t Ndim, std::floating_point TData>
inline auto PointsHost<Ndim, TData>::delta() const {
  assert(m_clustered && "The points have to be clustered before the deltas can be accessed");
  assert(has_details() && "store_details must be called before clustering to access delta");
  return std::as_const(m_view).delta();
}

template <std::size_t Ndim, std::floating_point TData>
inline auto PointsHost<Ndim, TData>::nearest_higher() const {
  assert(m_clustered &&
         "The points have to be clustered before the nearest highers can be accessed");
  assert(has_details() &&
         "store_details must be called before clustering to access nearest_higher");
  return std::as_const(m_view).nearest_higher();
}

template <std::size_t Ndim, std::floating_point TData>
inline void PointsHost<Ndim, TData>::set_density_uncertainty(
    std::span<const element_type> density_uncertainty) {
  if (density_uncertainty.size() != static_cast<size_t>(m_size)) {
    throw std::invalid_argument("Size of density_uncertainty does not match the number of points");
  }
  m_view.m_density_uncertainty = density_uncertainty.data();
}

template <std::size_t Ndim, std::floating_point TData>
inline void PointsHost<Ndim, TData>::set_sigma(std::size_t dim,
                                               std::span<const element_type> uncertainty) {
  if (dim >= Ndim) {
    throw std::out_of_range("Dimension out of range in set_sigma");
  }
  if (uncertainty.size() != static_cast<size_t>(m_size)) {
    throw std::invalid_argument("Size of sigma does not match the number of points");
  }
  m_view.m_sigmas[dim] = uncertainty.data();
}

template <std::size_t Ndim, std::floating_point TData>
inline void PointsHost<Ndim, TData>::set_tags(std::span<const std::uint32_t> tags) {
  if (tags.size() != static_cast<size_t>(m_size)) {
    throw std::invalid_argument("Size of tags does not match the number of points");
  }
  m_view.m_tags = tags.data();
}

CLUE_NAMESPACE_END
