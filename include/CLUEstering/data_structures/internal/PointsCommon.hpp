
#pragma once

#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <span>
#include <type_traits>

namespace clue {

  namespace internal {

    template <typename TPoints>
    struct points_interface {
      ALPAKA_FN_HOST auto size() const { return static_cast<const TPoints*>(this)->m_size; }

      ALPAKA_FN_HOST auto coords(std::size_t dim) const {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return view.coords()[dim];
      }
      ALPAKA_FN_HOST auto coords(std::size_t dim)
        requires std::same_as<typename TPoints::element_type,
                              std::remove_cv_t<typename TPoints::element_type>>
      {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<TPoints*>(this)->m_view;
        return view.coords()[dim];
      }

      ALPAKA_FN_HOST auto weights() const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return view.weights();
      }
      ALPAKA_FN_HOST auto weights()
        requires std::same_as<typename TPoints::element_type,
                              std::remove_cv_t<typename TPoints::element_type>>
      {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return view.weights();
      }

      ALPAKA_FN_HOST auto clusterIndexes() const {
        assert(static_cast<const TPoints&>(*this).m_clustered &&
               "The points have not been clustered yet, so the cluster indexes cannot be accessed");
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return view.cluster_index();
      }

      ALPAKA_FN_HOST auto clustered() const {
        return static_cast<const TPoints&>(*this).m_clustered;
      }

      ALPAKA_FN_HOST const auto& view() const { return static_cast<const TPoints*>(this)->m_view; }
      ALPAKA_FN_HOST auto& view() { return static_cast<TPoints*>(this)->m_view; }
    };

  }  // namespace internal

  template <std::size_t Ndim, std::floating_point TElement = float>
  struct PointsView {
    using element_type = TElement;
    using value_type = std::remove_cv_t<TElement>;

    std::array<element_type*, Ndim> m_coords;
    element_type* m_weight;
    std::int32_t* m_cluster_index;
    std::int32_t* m_is_seed;
    value_type* m_rho;
    std::int32_t* m_nearest_higher;
    std::int32_t m_n;

    ALPAKA_FN_HOST_ACC auto coords() const {
      std::array<std::span<const value_type>, Ndim> coord_spans;
      for (std::size_t dim = 0; dim < Ndim; ++dim) {
        coord_spans[dim] = std::span<const value_type>(m_coords[dim], m_n);
      }
      return coord_spans;
    }
    ALPAKA_FN_HOST_ACC auto coords() {
      std::array<std::span<value_type>, Ndim> coord_spans;
      for (std::size_t dim = 0; dim < Ndim; ++dim) {
        coord_spans[dim] = std::span<value_type>(m_coords[dim], m_n);
      }
      return coord_spans;
    }
    ALPAKA_FN_HOST_ACC auto weights() const { return std::span<const value_type>(m_weight, m_n); }
    ALPAKA_FN_HOST_ACC auto weights() {
      auto& view = *this;
      return std::span<value_type>(m_weight, m_n);
    }
    ALPAKA_FN_HOST_ACC auto cluster_index() const {
      assert(m_cluster_index != nullptr &&
             "The cluster indexes have not been allocated yet, so they cannot be accessed");
      return std::span<const int>(m_cluster_index, m_n);
    }
    ALPAKA_FN_HOST_ACC auto cluster_index() {
      assert(m_cluster_index != nullptr &&
             "The cluster indexes have not been allocated yet, so they cannot be accessed");
      return std::span<int>(m_cluster_index, m_n);
    }
    ALPAKA_FN_HOST_ACC auto is_seed() const {
      assert(m_is_seed != nullptr &&
             "The is_seed array has not been allocated yet, so it cannot be accessed");
      return std::span<const std::int32_t>(m_is_seed, m_n);
    }
    ALPAKA_FN_HOST_ACC auto is_seed() {
      assert(m_is_seed != nullptr &&
             "The is_seed array has not been allocated yet, so it cannot be accessed");
      return std::span<std::int32_t>(m_is_seed, m_n);
    }
    ALPAKA_FN_HOST_ACC auto rho() const {
      assert(m_rho != nullptr &&
             "The rho array has not been allocated yet, so it cannot be accessed");
      return std::span<const value_type>(m_rho, m_n);
    }
    ALPAKA_FN_HOST_ACC auto rho() {
      assert(m_rho != nullptr &&
             "The rho array has not been allocated yet, so it cannot be accessed");
      return std::span<value_type>(m_rho, m_n);
    }
    ALPAKA_FN_HOST_ACC auto nearest_higher() const {
      assert(m_nearest_higher != nullptr &&
             "The nearest_higher array has not been allocated yet, so it cannot be accessed");
      return std::span<const std::int32_t>(m_nearest_higher, m_n);
    }
    ALPAKA_FN_HOST_ACC auto nearest_higher() {
      assert(m_nearest_higher != nullptr &&
             "The nearest_higher array has not been allocated yet, so it cannot be accessed");
      return std::span<std::int32_t>(m_nearest_higher, m_n);
    }
    ALPAKA_FN_HOST_ACC auto size() const { return m_n; }

    ALPAKA_FN_HOST_ACC auto operator[](int index) const {
      if (index == -1)
        return clue::nostd::make_array<value_type, Ndim + 1>(
            std::numeric_limits<value_type>::max());

      std::array<value_type, Ndim + 1> point;
      meta::apply<Ndim>([&]<std::size_t Dim>() -> void { point[Dim] = m_coords[Dim][index]; });
      point[Ndim] = m_weight[index];
      return point;
    }
  };

  // TODO: implement for better cache use
  template <std::size_t Ndim>
  int32_t computeAlignSoASize(int32_t n_points);

}  // namespace clue
