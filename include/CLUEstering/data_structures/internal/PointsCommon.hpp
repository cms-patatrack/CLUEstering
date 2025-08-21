
#pragma once

#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <span>

namespace clue {

  namespace internal {

    template <typename TPoints>
    struct points_interface {
      ALPAKA_FN_HOST int32_t size() const { return static_cast<const TPoints*>(this)->m_size; }

      ALPAKA_FN_HOST auto coords() const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const float>(view.coords, view.n * TPoints::Ndim_);
      }
      ALPAKA_FN_HOST auto coords() {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<float>(view.coords, view.n * TPoints::Ndim_);
      }

      ALPAKA_FN_HOST auto coords(size_t dim) const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const float>(view.coords + dim * view.n, view.n);
      }
      ALPAKA_FN_HOST auto coords(size_t dim) {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<float>(view.coords + dim * view.n, view.n);
      }

      ALPAKA_FN_HOST auto weights() const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const float>(view.weight, view.n);
      }
      ALPAKA_FN_HOST auto weights() {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<float>(view.weight, view.n);
      }

      ALPAKA_FN_HOST auto clusterIndexes() const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const int>(view.cluster_index, view.n);
      }
      ALPAKA_FN_HOST auto clusterIndexes() {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<int>(view.cluster_index, view.n);
      }

      ALPAKA_FN_HOST auto isSeed() const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const int>(view.is_seed, view.n);
      }
      ALPAKA_FN_HOST auto isSeed() {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<int>(view.is_seed, view.n);
      }

      ALPAKA_FN_HOST const auto& view() const { return static_cast<const TPoints*>(this)->m_view; }
      ALPAKA_FN_HOST auto& view() { return static_cast<TPoints*>(this)->m_view; }
    };

  }  // namespace internal

  struct PointsView {
    float* coords;
    float* weight;
    int* cluster_index;
    int* is_seed;
    float* rho;
    float* delta;
    int* nearest_higher;
    int32_t n;
  };

  namespace concepts {

    template <typename T>
    concept contiguous_raw_data = std::is_array_v<T> || std::is_pointer_v<T>;

  }  // namespace concepts

  // TODO: implement for better cache use
  template <uint8_t Ndim>
  int32_t computeAlignSoASize(int32_t n_points);

  template <uint8_t Ndim>
  class PointsHost;
  template <uint8_t Ndim, concepts::device TDev>
  class PointsDevice;

  template <concepts::queue TQueue, uint8_t Ndim, concepts::device TDev>
  void copyToHost(TQueue& queue,
                  PointsHost<Ndim>& h_points,
                  const PointsDevice<Ndim, TDev>& d_points) {
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view.cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view.is_seed, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_view.is_seed, h_points.size()));
  }
  template <concepts::queue TQueue, uint8_t Ndim, concepts::device TDev>
  void copyToDevice(TQueue& queue,
                    PointsDevice<Ndim, TDev>& d_points,
                    const PointsHost<Ndim>& h_points) {
    alpaka::memcpy(
        queue,
        make_device_view(alpaka::getDev(queue), d_points.m_view.coords, Ndim * h_points.size()),
        make_host_view(h_points.m_view.coords, Ndim * h_points.size()));
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));
  }

}  // namespace clue
