
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <cstdint>
#include <span>

namespace clue {

  class AssociationMapView;
  template <concepts::device TDev>
  class AssociationMap;

  template <concepts::device TDev>
  class Followers {
  public:
    Followers(int32_t npoints, const TDev& dev) : m_assoc(npoints, npoints, dev) {}
    template <concepts::queue TQueue>
    Followers(int32_t npoints, TQueue& queue) : m_assoc(npoints, npoints, queue) {}

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(int32_t npoints, TQueue& queue) {
      m_assoc.initialize(npoints, npoints, queue);
    }
    ALPAKA_FN_HOST void reset(int32_t npoints) { m_assoc.reset(npoints, npoints); }

    template <concepts::accelerator TAcc, concepts::queue TQueue, std::size_t Ndim>
    ALPAKA_FN_HOST void fill(TQueue& queue, const PointsDevice<Ndim, TDev>& d_points) {
      m_assoc.template fill<TAcc>(
          d_points.size(),
          std::span<std::int32_t>{d_points.view().nearest_higher, d_points.size()},
          queue);
    }

    ALPAKA_FN_HOST inline constexpr int32_t extents() const { return m_assoc.extents().values; }

    ALPAKA_FN_HOST const AssociationMapView& view() const { return m_assoc.view(); }
    ALPAKA_FN_HOST AssociationMapView& view() { return m_assoc.view(); }

  private:
    AssociationMap<TDev> m_assoc;
  };

  using FollowersView = clue::AssociationMapView;

}  // namespace clue
