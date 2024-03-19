#ifndef Tiles_Alpaka_h
#define Tiles_Alpaka_h

#include <alpaka/core/Common.hpp>
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdint.h>
#include <type_traits>

#include "../../AlpakaCore/alpakaConfig.h"
#include "../../AlpakaCore/alpakaMemory.h"
#include "AlpakaVecArray.h"

using cms::alpakatools::VecArray;

constexpr uint32_t max_tile_depth{1 << 10};
constexpr uint32_t max_n_tiles{1 << 10};

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <uint8_t Ndim>
  class TilesAlpaka {
  public:
    TilesAlpaka()
        : n_tiles{1000}, n_tiles_per_dim{static_cast<int>(std::pow(1000, 1. / Ndim))} {};

    // Public member
    VecArray<VecArray<float, 2>, Ndim> min_max;
    VecArray<float, Ndim> tile_size;

    // Public methods
    void resizeTiles() { m_tiles.resize(n_tiles); }

    // getter
    int nPerDim() const { return n_tiles_per_dim; }

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC inline constexpr int getBin(const TAcc& acc,
                                                   float coord_,
                                                   int dim_) const {
      int coord_Bin{(int)((coord_ - min_max[dim_][0]) / tile_size[dim_])};

      // Address the cases of underflow and overflow and underflow
      coord_Bin = alpaka::math::min(acc, coord_Bin, n_tiles_per_dim - 1);
      coord_Bin = alpaka::math::max(acc, coord_Bin, 0);

      return coord_Bin;
    }

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC inline constexpr int getGlobalBin(
        const TAcc& acc, const VecArray<float, Ndim>& coords) const {
      int globalBin{getBin(acc, coords[0], 0)};
      for (int i{1}; i != Ndim; ++i) {
        globalBin += n_tiles_per_dim * getBin(acc, coords[i], i);
      }
      return globalBin;
    }

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC inline constexpr int getGlobalBinByBin(
        const TAcc& acc, const VecArray<uint32_t, Ndim>& Bins) const {
      uint32_t globalBin{Bins[0]};
      for (int i{1}; i != Ndim; ++i) {
        globalBin += n_tiles_per_dim * Bins[i];
      }
      return globalBin;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC inline constexpr void fill(const TAcc& acc,
                                             const VecArray<float, Ndim>& coords,
                                             int i) {
      m_tiles[getGlobalBin(acc, coords)].push_back(acc, i);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC inline void searchBox(
        const TAcc& acc,
        const VecArray<VecArray<float, 2>, Ndim>& sb_extremes,
        VecArray<VecArray<uint32_t, 2>, Ndim>* search_box) {
      for (int dim{}; dim != Ndim; ++dim) {
        VecArray<uint32_t, 2> dim_sb;
        dim_sb.push_back_unsafe(getBin(acc, sb_extremes[dim][0], dim));
        dim_sb.push_back_unsafe(getBin(acc, sb_extremes[dim][1], dim));

        search_box->push_back_unsafe(dim_sb);
      }
    }

    ALPAKA_FN_HOST_ACC inline constexpr auto size() { return n_tiles; }

    ALPAKA_FN_HOST_ACC inline constexpr void clear() {
      for (int i{}; i < n_tiles; ++i) {
        m_tiles[i].reset();
      }
    }

    ALPAKA_FN_HOST_ACC inline constexpr VecArray<uint32_t, max_tile_depth>& operator[](
        int globalBinId) {
      return m_tiles[globalBinId];
    }

  private:
    size_t n_tiles;
    int n_tiles_per_dim;
    VecArray<VecArray<uint32_t, max_tile_depth>, max_n_tiles> m_tiles;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
