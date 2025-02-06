
#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <stdint.h>

#include "../../AlpakaCore/alpakaWorkDiv.hpp"
#include "../../AlpakaCore/alpakaConfig.hpp"
#include "../../AlpakaCore/alpakaMemory.hpp"
#include "AlpakaVecArray.hpp"
#include "AssociationMap.hpp"

using clue::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

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
    uint32_t* indexes;
    uint32_t* offsets;
    CoordinateExtremes<Ndim>* minmax;
    float* tilesizes;
    int32_t* wrapped;
    uint32_t npoints;
    int32_t ntiles;
    int32_t nperdim;

    ALPAKA_FN_ACC inline constexpr const float* minMax() const { return minmax; }
    ALPAKA_FN_ACC inline constexpr float* minMax() { return minmax; }

    ALPAKA_FN_ACC inline constexpr const float* tileSize() const { return tilesizes; }
    ALPAKA_FN_ACC inline constexpr float* tileSize() { return tilesizes; }

    ALPAKA_FN_ACC inline constexpr const int* wrapped() const { return wrapped; }
    ALPAKA_FN_ACC inline constexpr int* wrapped() { return wrapped; }

    template <typename TAcc>
    ALPAKA_FN_ACC inline constexpr int getBin(const TAcc& acc,
                                              float coord,
                                              int dim) const {
      int coord_bin;
      if (m_wrapped[dim]) {
        coord_bin = static_cast<int>(
            (normalizeCoordinate(coord, dim) - minmax->min(dim)) / tile_size[dim]);
      } else {
        coord_bin = static_cast<int>((coord - minmax->min(dim)) / tile_size[dim]);
      }

      // Address the cases of underflow and overflow
      coord_bin = alpaka::math::min(acc, coord_bin, nperdim - 1);
      coord_bin = alpaka::math::max(acc, coord_bin, 0);

      return coord_bin;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC inline constexpr int getGlobalBin(const TAcc& acc,
                                                    const float* coords) const {
      int global_bin = 0;
      for (int dim = 0; dim != Ndim - 1; ++dim) {
        global_bin += alpaka::math::pow(acc, nperdim, Ndim - dim - 1) *
                      getBin(acc, coords[dim], dim);
      }
      global_bin += getBin(acc, coords[Ndim - 1], Ndim - 1);
      return global_bin;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC inline constexpr int getGlobalBinByBin(
        const TAcc& acc, const VecArray<uint32_t, Ndim>& Bins) const {
      uint32_t global_bin = 0;
      for (int dim = 0; dim != Ndim; ++dim) {
        auto bin_i = m_wrapped[dim] ? (Bins[dim] % nperdim) : Bins[dim];
        globalBin += alpaka::math::pow(acc, nperdim, Ndim - dim - 1) * bin_i;
      }
      return globalBin;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC inline void searchBox(
        const TAcc& acc,
        const VecArray<VecArray<float, 2>, Ndim>& sb_extremes,
        VecArray<VecArray<uint32_t, 2>, Ndim>* search_box) {
      for (int dim{}; dim != Ndim; ++dim) {
        VecArray<uint32_t, 2> dim_sb;
        auto infBin = getBin(acc, sb_extremes[dim][0], dim);
        auto supBin = getBin(acc, sb_extremes[dim][1], dim);
        if (m_wrapped[dim] and infBin > supBin)
          supBin += nperdim;
        dim_sb.push_back_unsafe(infBin);
        dim_sb.push_back_unsafe(supBin);

        search_box->push_back_unsafe(dim_sb);
      }
    }

    ALPAKA_FN_ACC inline constexpr clue::Span<uint32_t> operator[](uint32_t globalBinId) {
      const auto size = offsets[globalBinId + 1] - offsets[globalBinId];
      const auto offset = offsets[globalBinId];
      uint32_t* buf_ptr = indexes + offset;
      return clue::Span<uint32_t>{buf_ptr, size};
    }

    ALPAKA_FN_HOST_ACC inline constexpr float normalizeCoordinate(float coord,
                                                                  int dim) const {
      const float range = minmax->range(dim);
      float remainder = coord - static_cast<int>(coord / range) * range;
      if (remainder >= minmax->max(dim))
        remainder -= range;
      else if (remainder < minmax->min(dim))
        remainder += range;
      return remainder;
    }

    ALPAKA_FN_ACC inline float distance(const float* coord_i, const float* coord_j) {
      float dist_sq = 0.f;
      for (int dim = 0; dim != Ndim; ++dim) {
        if (m_wrapped[dim])
          dist_sq += normalizeCoordinate(coord_i[dim] - coord_j[dim], dim) *
                     normalizeCoordinate(coord_i[dim] - coord_j[dim], dim);
        else
          dist_sq += (coord_i[dim] - coord_j[dim]) * (coord_i[dim] - coord_j[dim]);
      }
      return dist_sq;
    }
  };

  template <uint8_t Ndim>
  class TilesAlpaka {
  public:
    TilesAlpaka(Queue queue, uint32_t n_points, int32_t n_perdim, int32_t n_tiles)
        : m_assoc{clue::AssociationMap<Device>(n_points, n_tiles, queue)},
          m_minmax{clue::make_device_buffer<CoordinateExtremes<Ndim>>(queue)},
          m_tilesizes{clue::make_device_buffer<float[Ndim]>(queue)},
          m_wrapped{clue::make_device_buffer<int[Ndim]>(queue)},
          m_ntiles{n_tiles},
          m_nperdim{n_perdim},
          m_view{clue::make_device_buffer<TilesAlpakaView<Ndim>>(queue)} {
      auto host_view = clue::make_host_buffer<TilesAlpakaView<Ndim>>(queue);
      host_view->indexes = m_assoc.indexes().data();
      host_view->offsets = m_assoc.offsets().data();
      host_view->minmax = m_minmax.data();
      host_view->tilesizes = m_tilesizes.data();
      host_view->wrapped = m_wrapped.data();
      host_view->npoints = n_points;
      host_view->ntiles = n_tiles;
      host_view->nperdim = n_perdim;

      alpaka::memcpy(queue, m_view, host_view);
    }

    TilesAlpakaView<Ndim>* view() { return m_view.data(); }

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    ALPAKA_FN_HOST void initialize(uint32_t size, uint32_t nbins, const TQueue& queue) {
      m_assoc.initialize(size, nbins, queue);
    }

    struct GetGlobalBin {
      PointsAlpakaView* pointsView;
      TilesAlpakaView<Ndim>* tilesView;

      template <typename TAcc>
      ALPAKA_FN_ACC uint32_t operator()(const TAcc& acc, uint32_t index) const {
        float coords[Ndim];
        for (auto dim = 0; dim < Ndim; ++dim) {
          coords[dim] = pointsView->coords[index + dim * pointsView->n];
        }

        auto bin = tilesView->getGlobalBin(acc, coords);
        return bin;
      }
    };

    ALPAKA_FN_HOST void fill(Queue queue, PointsAlpaka<Ndim>& d_points, size_t size) {
      auto dev = alpaka::getDev(queue);
      auto pointsView = d_points.view();
      m_assoc.fill<Acc1D>(size, GetGlobalBin{pointsView, m_view.data()}, queue);
    }

    ALPAKA_FN_HOST inline clue::device_buffer<Device, CoordinateExtremes<Ndim>> minMax()
        const {
      return m_minmax;
    }
    ALPAKA_FN_HOST inline clue::device_buffer<Device, float[Ndim]> tileSize() const {
      return m_tilesizes;
    }
    ALPAKA_FN_HOST inline clue::device_buffer<Device, int32_t[Ndim]> wrapped() const {
      return m_wrapped;
    }

    ALPAKA_FN_HOST inline constexpr auto size() { return m_ntiles; }

    ALPAKA_FN_HOST inline constexpr auto nPerDim() const { return m_nperdim; }

    ALPAKA_FN_HOST inline constexpr void clear(const Queue& queue) {}

    ALPAKA_FN_HOST clue::device_view<Device, uint32_t[]> indexes(const Device& dev,
                                                                 size_t assoc_id) {
      return m_assoc.indexes(dev, assoc_id);
    }

  private:

    clue::AssociationMap<Device> m_assoc;
    clue::device_buffer<Device, CoordinateExtremes<Ndim>> m_minmax;
    clue::device_buffer<Device, float[Ndim]> m_tilesizes;
    clue::device_buffer<Device, int32_t[Ndim]> m_wrapped;
    int32_t m_ntiles;
    int32_t m_nperdim;
    clue::device_buffer<Device, TilesAlpakaView<Ndim>> m_view;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
