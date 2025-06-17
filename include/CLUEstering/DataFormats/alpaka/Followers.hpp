
#pragma once

#include "CLUEstering/DataFormats/alpaka/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"

namespace clue {

  namespace concepts = detail::concepts;

  class AssociationMapView;
  template <concepts::device TDev>
  class AssociationMap;

  template <concepts::device TDev>
  class Followers {
  public:
    Followers(uint32_t npoints, const TDev& dev)
        : m_assoc(npoints, npoints, dev), m_view{make_device_buffer<AssociationMapView>(dev)} {}
    template <concepts::queue TQueue>
    Followers(uint32_t npoints, TQueue& queue)
        : m_assoc(npoints, npoints, queue), m_view{make_device_buffer<AssociationMapView>(queue)} {}

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(uint32_t npoints, TQueue& queue) {
      m_assoc.initialize(npoints, npoints, queue);
    }
    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void reset(uint32_t npoints, TQueue& queue) {
      m_assoc.reset(queue, npoints, npoints);
    }

    template <concepts::accelerator TAcc, concepts::queue TQueue, uint8_t Ndim>
    ALPAKA_FN_HOST void fill(TQueue& queue, PointsDevice<Ndim, TDev>& d_points) {
      m_assoc.template fill<TAcc>(d_points.size(), d_points.nearestHigher(), queue);
    }

    ALPAKA_FN_HOST inline constexpr uint32_t extents() const { return m_assoc.extents().content; }

    ALPAKA_FN_HOST AssociationMapView* view() { return m_assoc.view(); }

  private:
    AssociationMap<TDev> m_assoc;
    device_buffer<TDev, AssociationMapView> m_view;
  };

  using FollowersView = clue::AssociationMapView;

}  // namespace clue
