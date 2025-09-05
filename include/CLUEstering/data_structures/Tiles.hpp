
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/VecArray.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <uint8_t Ndim>
  class CoordinateExtremes {
  private:
    float m_data[2 * Ndim];

  public:
    CoordinateExtremes() = default;

    ALPAKA_FN_HOST_ACC const float* data() const { return m_data; }
    ALPAKA_FN_HOST_ACC float* data() { return m_data; }

    ALPAKA_FN_HOST_ACC float min(int i) const { return m_data[2 * i]; }
    ALPAKA_FN_HOST_ACC float& min(int i) { return m_data[2 * i]; }
    ALPAKA_FN_HOST_ACC float max(int i) const { return m_data[2 * i + 1]; }
    ALPAKA_FN_HOST_ACC float& max(int i) { return m_data[2 * i + 1]; }
    ALPAKA_FN_HOST_ACC float range(int i) const { return max(i) - min(i); }
  };

  template <uint8_t Ndim>
  struct TilesAlpakaView {
    int32_t* indexes;
    int32_t* offsets;
    CoordinateExtremes<Ndim>* minmax;
    float* tilesizes;
    uint8_t* wrapping;
    int32_t npoints;
    int32_t ntiles;
    int32_t nperdim;

    ALPAKA_FN_ACC inline constexpr const float* minMax() const { return minmax; }
    ALPAKA_FN_ACC inline constexpr float* minMax() { return minmax; }

    ALPAKA_FN_ACC inline constexpr const float* tileSize() const { return tilesizes; }
    ALPAKA_FN_ACC inline constexpr float* tileSize() { return tilesizes; }

    ALPAKA_FN_ACC inline constexpr const uint8_t* wrapped() const { return wrapping; }
    ALPAKA_FN_ACC inline constexpr uint8_t* wrapped() { return wrapping; }

    ALPAKA_FN_ACC inline constexpr int getBin(float coord, int dim) const {
      int coord_bin;
      if (wrapping[dim]) {
        coord_bin =
            static_cast<int>((normalizeCoordinate(coord, dim) - minmax->min(dim)) / tilesizes[dim]);
      } else {
        coord_bin = static_cast<int>((coord - minmax->min(dim)) / tilesizes[dim]);
      }

      // Address the cases of underflow and overflow
      coord_bin = internal::math::min(coord_bin, nperdim - 1);
      coord_bin = internal::math::max(coord_bin, 0);

      return coord_bin;
    }

    ALPAKA_FN_ACC inline constexpr int getGlobalBin(const float* coords) const {
      int global_bin = 0;
      for (int dim = 0; dim != Ndim - 1; ++dim) {
        global_bin += internal::math::pow(static_cast<float>(nperdim), Ndim - dim - 1) *
                      getBin(coords[dim], dim);
      }
      global_bin += getBin(coords[Ndim - 1], Ndim - 1);
      return global_bin;
    }

    ALPAKA_FN_ACC inline constexpr int getGlobalBinByBin(const VecArray<int32_t, Ndim>& Bins) const {
      int32_t globalBin = 0;
      for (int dim = 0; dim != Ndim; ++dim) {
        auto bin_i = wrapping[dim] ? (Bins[dim] % nperdim) : Bins[dim];
        globalBin += internal::math::pow(static_cast<float>(nperdim), Ndim - dim - 1) * bin_i;
      }
      return globalBin;
    }

    ALPAKA_FN_ACC inline void searchBox(const SearchBoxExtremes<Ndim>& searchbox_extremes,
                                        SearchBoxBins<Ndim>& searchbox_bins) {
      for (int dim{}; dim != Ndim; ++dim) {
        auto infBin = getBin(searchbox_extremes[dim][0], dim);
        auto supBin = getBin(searchbox_extremes[dim][1], dim);
        if (wrapping[dim] and infBin > supBin)
          supBin += nperdim;

        searchbox_bins[dim] = nostd::make_array(infBin, supBin);
      }
    }

    ALPAKA_FN_ACC inline constexpr clue::Span<int32_t> operator[](int32_t globalBinId) {
      const auto size = offsets[globalBinId + 1] - offsets[globalBinId];
      const auto offset = offsets[globalBinId];
      int32_t* buf_ptr = indexes + offset;
      return clue::Span<int32_t>{buf_ptr, size};
    }

    ALPAKA_FN_HOST_ACC inline constexpr float normalizeCoordinate(float coord, int dim) const {
      const float range = minmax->range(dim);
      float remainder = coord - static_cast<int>(coord / range) * range;
      if (remainder >= minmax->max(dim))
        remainder -= range;
      else if (remainder < minmax->min(dim))
        remainder += range;
      return remainder;
    }

    ALPAKA_FN_ACC inline float distance(const std::array<float, Ndim>& coord_i,
                                        const std::array<float, Ndim>& coord_j) {
      float dist_sq = 0.f;
      for (int dim = 0; dim != Ndim; ++dim) {
        if (wrapping[dim])
          dist_sq += normalizeCoordinate(coord_i[dim] - coord_j[dim], dim) *
                     normalizeCoordinate(coord_i[dim] - coord_j[dim], dim);
        else
          dist_sq += (coord_i[dim] - coord_j[dim]) * (coord_i[dim] - coord_j[dim]);
      }
      return dist_sq;
    }
  };

  template <uint8_t Ndim, concepts::device TDev>
  class TilesAlpaka {
  public:
    template <concepts::queue TQueue>
    TilesAlpaka(TQueue& queue, int32_t n_points, int32_t n_tiles)
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

    const TilesAlpakaView<Ndim>& view() const { return m_view; }
    TilesAlpakaView<Ndim>& view() { return m_view; }

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(int32_t npoints, int32_t ntiles, int32_t nperdim, TQueue& queue) {
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

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void reset(int32_t npoints, int32_t ntiles, int32_t nperdim, TQueue& queue) {
      m_assoc.reset(queue, npoints, ntiles);

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
      PointsView pointsView;
      TilesAlpakaView<Ndim> tilesView;

      ALPAKA_FN_ACC int32_t operator()(int32_t index) const {
        float coords[Ndim];
        for (auto dim = 0; dim < Ndim; ++dim) {
          coords[dim] = pointsView.coords[index + dim * pointsView.n];
        }

        auto bin = tilesView.getGlobalBin(coords);
        return bin;
      }
    };

    template <concepts::accelerator TAcc, concepts::queue TQueue>
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

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST inline constexpr void clear(const TQueue& queue) {}

    ALPAKA_FN_HOST const clue::device_buffer<TDev, int32_t[]>& indexes() const {
      return m_assoc.indexes();
    }
    ALPAKA_FN_HOST clue::device_buffer<TDev, int32_t[]>& indexes() { return m_assoc.indexes(); }
    ALPAKA_FN_HOST const clue::device_buffer<TDev, int32_t[]>& offsets() const {
      return m_assoc.offsets();
    }
    ALPAKA_FN_HOST clue::device_buffer<TDev, int32_t[]>& offsets() { return m_assoc.offsets(); }

    ALPAKA_FN_HOST clue::device_view<TDev, int32_t[]> indexes(const TDev& dev, size_t assoc_id) {
      return m_assoc.indexes(dev, assoc_id);
    }

  private:
    AssociationMap<TDev> m_assoc;
    device_buffer<TDev, CoordinateExtremes<Ndim>> m_minmax;
    device_buffer<TDev, float[Ndim]> m_tilesizes;
    device_buffer<TDev, uint8_t[Ndim]> m_wrapped;
    int32_t m_ntiles;
    int32_t m_nperdim;
    TilesAlpakaView<Ndim> m_view;
  };

}  // namespace clue
