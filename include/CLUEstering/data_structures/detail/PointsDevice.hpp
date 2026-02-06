
#pragma once

#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/detail/DeviceViewPartition.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/reduce/reduce.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"

#include <alpaka/alpaka.hpp>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>

namespace clue {

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TData, TDev>::PointsDevice(TQueue& queue, int32_t n_points)
      : m_buffer{make_device_buffer<std::byte[]>(
            queue, soa::device::computeSoASize<Ndim, value_type>(n_points))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points);
  }

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TData, TDev>::PointsDevice(TQueue& queue,
                                                       int32_t n_points,
                                                       std::span<std::byte> buffer)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(value_type))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), buffer.data(), n_points);
  }

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TData, TDev>::PointsDevice(TQueue& queue,
                                                       int32_t n_points,
                                                       std::span<value_type> input,
                                                       std::span<int> output)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(value_type))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points, input, output);
  }

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TData, TDev>::PointsDevice(TQueue& queue,
                                                       int32_t n_points,
                                                       std::span<value_type> coordinates,
                                                       std::span<value_type> weights,
                                                       std::span<int> output)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(value_type))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(
        m_view, m_buffer.data(), n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TData, TDev>::PointsDevice(TQueue& queue,
                                                       int32_t n_points,
                                                       value_type* input,
                                                       int* output)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(value_type))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points, input, output);
  }

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  template <concepts::queue TQueue>
  inline PointsDevice<Ndim, TData, TDev>::PointsDevice(
      TQueue& queue, int32_t n_points, value_type* coordinates, value_type* weights, int* output)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(value_type))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(
        m_view, m_buffer.data(), n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  template <concepts::queue TQueue, concepts::pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
  inline PointsDevice<Ndim, TData, TDev>::PointsDevice(TQueue& queue,
                                                       int32_t n_points,
                                                       TBuffers... buffers)
      : m_buffer{make_device_buffer<std::byte[]>(queue, 3 * n_points * sizeof(value_type))},
        m_view{},
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points, buffers...);
  }

  template <std::size_t Ndim, std::floating_point TData, concepts::device TDev>
  ALPAKA_FN_HOST inline const auto& PointsDevice<Ndim, TData, TDev>::n_clusters() {
    assert(m_clustered &&
           "The points have to be clustered before the cluster properties can be accessed");
    if (!m_nclusters.has_value()) {
      auto cluster_ids = this->clusterIndexes();
      m_nclusters = internal::algorithm::reduce(cluster_ids.begin(),
                                                cluster_ids.end(),
                                                std::numeric_limits<int32_t>::lowest(),
                                                clue::nostd::maximum<int32_t>{}) +
                    1;
    }

    return m_nclusters.value();
  }

}  // namespace clue
