
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <alpaka/alpaka.hpp>

namespace clue::internal {

  template <std::size_t Ndim, clue::concepts::device TDev>
  class Tiles {
  public:
    template <clue::concepts::queue TQueue>
    Tiles(TQueue& queue, int32_t n_points, int32_t n_tiles)
        : m_assoc{AssociationMap<TDev>(n_points, n_tiles, queue)},
          m_minmax{make_device_buffer<CoordinateExtremes<Ndim>>(queue)},
          m_tilesizes{make_device_buffer<float[Ndim]>(queue)},
          m_wrapped{make_device_buffer<uint8_t[Ndim]>(queue)},
          m_ntiles{n_tiles},
          m_nperdim{static_cast<int32_t>(std::pow(n_tiles, 1.f / Ndim))},
          m_view{} {
      m_view.indexes = m_assoc.indexes().data();
      m_view.offsets = m_assoc.offsets().data();
      m_view.minmax = m_minmax.data();
      m_view.tilesizes = m_tilesizes.data();
      m_view.wrapping = m_wrapped.data();
      m_view.npoints = n_points;
      m_view.ntiles = m_ntiles;
      m_view.nperdim = m_nperdim;
    }

    const TilesView<Ndim>& view() const { return m_view; }
    TilesView<Ndim>& view() { return m_view; }

    template <clue::concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(TQueue& queue, int32_t npoints, int32_t ntiles, int32_t nperdim) {
      m_assoc.initialize(npoints, ntiles, queue);
      m_ntiles = ntiles;
      m_nperdim = nperdim;

      m_view.indexes = m_assoc.indexes().data();
      m_view.offsets = m_assoc.offsets().data();
      m_view.minmax = m_minmax.data();
      m_view.tilesizes = m_tilesizes.data();
      m_view.wrapping = m_wrapped.data();
      m_view.npoints = npoints;
      m_view.ntiles = ntiles;
      m_view.nperdim = nperdim;
    }

    ALPAKA_FN_HOST void reset(int32_t npoints, int32_t ntiles, int32_t nperdim) {
      m_assoc.reset(npoints, ntiles);

      m_ntiles = ntiles;
      m_nperdim = nperdim;
      m_view.indexes = m_assoc.indexes().data();
      m_view.offsets = m_assoc.offsets().data();
      m_view.minmax = m_minmax.data();
      m_view.tilesizes = m_tilesizes.data();
      m_view.wrapping = m_wrapped.data();
      m_view.npoints = npoints;
      m_view.ntiles = ntiles;
      m_view.nperdim = nperdim;
    }

    struct GetGlobalBin {
      PointsView<Ndim> pointsView;
      TilesView<Ndim> tilesView;

      ALPAKA_FN_ACC int32_t operator()(int32_t index) const {
        float coords[Ndim];
        for (auto dim = 0u; dim < Ndim; ++dim) {
          coords[dim] = pointsView.coords[dim][index];
        }

        auto bin = tilesView.getGlobalBin(coords);
        return bin;
      }
    };

    template <clue::concepts::accelerator TAcc, clue::concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(TQueue& queue, PointsDevice<Ndim, TDev>& d_points, size_t size) {
      auto dev = alpaka::getDev(queue);
      auto pointsView = d_points.view();
      m_assoc.template fill<TAcc>(size, GetGlobalBin{pointsView, m_view}, queue);
    }

    ALPAKA_FN_HOST inline clue::device_buffer<TDev, CoordinateExtremes<Ndim>> minMax() const {
      return m_minmax;
    }
    ALPAKA_FN_HOST inline clue::device_buffer<TDev, float[Ndim]> tileSize() const {
      return m_tilesizes;
    }
    ALPAKA_FN_HOST inline clue::device_buffer<TDev, uint8_t[Ndim]> wrapped() const {
      return m_wrapped;
    }

    ALPAKA_FN_HOST inline constexpr auto size() const { return m_ntiles; }

    ALPAKA_FN_HOST inline constexpr auto nPerDim() const { return m_nperdim; }

    ALPAKA_FN_HOST inline constexpr auto extents() const { return m_assoc.extents(); }

  private:
    AssociationMap<TDev> m_assoc;
    device_buffer<TDev, CoordinateExtremes<Ndim>> m_minmax;
    device_buffer<TDev, float[Ndim]> m_tilesizes;
    device_buffer<TDev, uint8_t[Ndim]> m_wrapped;
    int32_t m_ntiles;
    int32_t m_nperdim;
    TilesView<Ndim> m_view;
  };

}  // namespace clue::internal
