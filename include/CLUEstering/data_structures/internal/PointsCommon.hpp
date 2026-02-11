
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
        return std::span<const typename TPoints::value_type>(view.coords[dim], view.n);
      }
      ALPAKA_FN_HOST auto coords(std::size_t dim)
        requires std::same_as<typename TPoints::element_type,
                              std::remove_cv_t<typename TPoints::element_type>>
      {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<typename TPoints::value_type>(view.coords[dim], view.n);
      }

      ALPAKA_FN_HOST auto weights() const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const typename TPoints::value_type>(view.weight, view.n);
      }
      ALPAKA_FN_HOST auto weights()
        requires std::same_as<typename TPoints::element_type,
                              std::remove_cv_t<typename TPoints::element_type>>
      {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<typename TPoints::value_type>(view.weight, view.n);
      }

      ALPAKA_FN_HOST auto clusterIndexes() const {
        assert(static_cast<const TPoints&>(*this).m_clustered &&
               "The points have not been clustered yet, so the cluster indexes cannot be accessed");
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const int>(view.cluster_index, view.n);
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

    std::array<element_type*, Ndim> coords;
    element_type* weight;
    std::int32_t* cluster_index;
    std::int32_t* is_seed;
    value_type* rho;
    std::int32_t* nearest_higher;
    std::int32_t n;

    ALPAKA_FN_HOST_ACC auto operator[](int index) const {
      if (index == -1)
        return clue::nostd::make_array<value_type, Ndim + 1>(
            std::numeric_limits<value_type>::max());

      std::array<value_type, Ndim + 1> point;
      meta::apply<Ndim>([&]<std::size_t Dim>() -> void { point[Dim] = coords[Dim][index]; });
      point[Ndim] = weight[index];
      return point;
    }
  };

  // TODO: implement for better cache use
  template <std::size_t Ndim>
  int32_t computeAlignSoASize(int32_t n_points);

}  // namespace clue
