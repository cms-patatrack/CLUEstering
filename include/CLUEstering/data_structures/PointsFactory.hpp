/// @file PointsFactory.hpp
/// @brief Provides factory functions for creating clustered points data structures
/// @authors Simone Balducci

#pragma once

#include "CLUEstering/detail/concepts.hpp"

#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>

namespace clue {

  /// @brief Factory function to create a clustered points data structure from raw pointers
  /// @param queue The queue to use for memory allocation
  /// @param n_points The number of points
  /// @param coordinates The pre-allocated buffer containing the coordinates
  /// @param weights The pre-allocated buffer containing the weights
  /// @param cluster_indexes The pre-allocated buffer to store the cluster indexes
  template <std::size_t Ndim, std::floating_point InputType, concepts::queue QueueType>
  auto make_clustered_points(QueueType& queue,
                             std::size_t n_points,
                             const InputType* coordinates,
                             const InputType* weights,
                             const int* cluster_indexes);

  /// @brief Factory function to create a clustered points data structure from spans
  /// @param queue The queue to use for memory allocation
  /// @param coordinates The pre-allocated buffer containing the coordinates
  /// @param weights The pre-allocated buffer containing the weights
  /// @param cluster_indexes The pre-allocated buffer to store the cluster indexes
  template <std::size_t Ndim, std::floating_point InputType, concepts::queue QueueType>
  auto make_clustered_points(QueueType& queue,
                             std::span<const InputType> coordinates,
                             std::span<const InputType> weights,
                             std::span<const int> cluster_indexes);

  /// @brief Factory function to create a clustered points data structure from raw pointers
  /// @param device The device to use for memory allocation
  /// @param n_points The number of points
  /// @param coordinates The pre-allocated buffer containing the coordinates
  /// @param weights The pre-allocated buffer containing the weights
  /// @param cluster_indexes The pre-allocated buffer to store the cluster indexes
  template <std::size_t Ndim, std::floating_point InputType, concepts::device DeviceType>
  auto make_clustered_points(const DeviceType& device,
                             std::size_t n_points,
                             const InputType* coordinates,
                             const InputType* weights,
                             const int* cluster_indexes);

  /// @brief Factory function to create a clustered points data structure from spans
  /// @param device The device to use for memory allocation
  /// @param coordinates The pre-allocated buffer containing the coordinates
  /// @param weights The pre-allocated buffer containing the weights
  /// @param cluster_indexes The pre-allocated buffer to store the cluster indexes
  template <std::size_t Ndim, std::floating_point InputType, concepts::device DeviceType>
  auto make_clustered_points(const DeviceType& device,
                             std::span<const InputType> coordinates,
                             std::span<const InputType> weights,
                             std::span<const int> cluster_indexes);

  /// @brief Factory function to create a clustered points data structure from per-dimension raw pointers
  /// @param queue The queue to use for memory allocation
  /// @param n_points The number of points
  /// @param buffers Ndim coordinate pointers, one weight pointer, and one cluster-index pointer
  template <std::size_t Ndim, concepts::queue QueueType, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(QueueType& queue, std::size_t n_points, TBuffers... buffers);

  /// @brief Factory function to create a clustered points data structure from per-dimension spans
  /// @param queue The queue to use for memory allocation
  /// @param buffers Ndim coordinate spans, one weight span, and one cluster-index span
  template <std::size_t Ndim, concepts::queue QueueType, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(QueueType& queue, TBuffers&&... buffers);

  /// @brief Factory function to create a clustered points data structure from per-dimension raw pointers
  /// @param device The device to use for memory allocation
  /// @param n_points The number of points
  /// @param buffers Ndim coordinate pointers, one weight pointer, and one cluster-index pointer
  template <std::size_t Ndim, concepts::device DeviceType, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(const DeviceType& device, std::size_t n_points, TBuffers... buffers);

  /// @brief Factory function to create a clustered points data structure from per-dimension spans
  /// @param device The device to use for memory allocation
  /// @param buffers Ndim coordinate spans, one weight span, and one cluster-index span
  template <std::size_t Ndim, concepts::device DeviceType, std::ranges::contiguous_range... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  auto make_clustered_points(const DeviceType& device, TBuffers&&... buffers);

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsFactory.hpp"
