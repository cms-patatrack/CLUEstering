
#pragma once

#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include <span>

namespace clue {

  namespace internal {

    template <typename TPoints>
    struct points_interface {
      ALPAKA_FN_HOST int32_t size() const { return static_cast<const TPoints*>(this)->m_size; }

      ALPAKA_FN_HOST auto coords(std::size_t dim) const {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const float>(view.coords[dim], view.n);
      }
      ALPAKA_FN_HOST auto coords(std::size_t dim) {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<float>(view.coords[dim], view.n);
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
        assert(static_cast<const TPoints&>(*this).m_clustered &&
               "The points have not been clustered yet, so the cluster indexes cannot be accessed");
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const int>(view.cluster_index, view.n);
      }
      ALPAKA_FN_HOST auto clusterIndexes() {
        assert(static_cast<const TPoints&>(*this).m_clustered &&
               "The points have not been clustered yet, so the cluster indexes cannot be accessed");
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<int>(view.cluster_index, view.n);
      }

      ALPAKA_FN_HOST auto clustered() const {
        return static_cast<const TPoints&>(*this).m_clustered;
      }

      ALPAKA_FN_HOST const auto& view() const { return static_cast<const TPoints*>(this)->m_view; }
      ALPAKA_FN_HOST auto& view() { return static_cast<TPoints*>(this)->m_view; }
    };

  }  // namespace internal

  template <std::size_t Ndim>
  struct PointsView {
    std::array<float*, Ndim> coords;
    float* weight;
    int* cluster_index;
    int* is_seed;
    float* rho;
    int* nearest_higher;
    int32_t n;

    ALPAKA_FN_HOST_ACC auto operator[](int i) const {
      if (i == -1)
        return clue::nostd::make_array<float, Ndim + 1>(std::numeric_limits<float>::max());

      std::array<float, Ndim + 1> point;
      meta::apply<Ndim>([&]<std::size_t Dim>() { point[Dim] = coords[Dim][i]; });
      point[Ndim] = weight[i];
      return point;
    }
  };

  // TODO: implement for better cache use
  template <std::size_t Ndim>
  int32_t computeAlignSoASize(int32_t n_points);

  template <std::size_t Ndim>
  class PointsHost;
  template <std::size_t Ndim, concepts::device TDev>
  class PointsDevice;

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToHost(TQueue& queue,
                  PointsHost<Ndim>& h_points,
                  const PointsDevice<Ndim, TDev>& d_points) {
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view.cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
    h_points.mark_clustered();
  }
  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToDevice(TQueue& queue,
                    PointsDevice<Ndim, TDev>& d_points,
                    const PointsHost<Ndim>& h_points) {
    // TODO: copy each coordinate column separately
    alpaka::memcpy(
        queue,
        make_device_view(alpaka::getDev(queue), d_points.m_view.coords[0], Ndim * h_points.size()),
        make_host_view(h_points.m_view.coords[0], Ndim * h_points.size()));
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));
  }

}  // namespace clue
