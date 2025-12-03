
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/VecArray.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/math/math.hpp"
#include <array>
#include <cstdint>
#include <alpaka/alpaka.hpp>

namespace clue::internal {

  template <std::size_t Ndim>
  struct TilesView {
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
      coord_bin = math::min(coord_bin, nperdim - 1);
      coord_bin = math::max(coord_bin, 0);

      return coord_bin;
    }

    ALPAKA_FN_ACC inline constexpr int getGlobalBin(const float* coords) const {
      int global_bin = 0;
      for (auto dim = 0u; dim != Ndim - 1; ++dim) {
        global_bin +=
            math::pow(static_cast<float>(nperdim), Ndim - dim - 1) * getBin(coords[dim], dim);
      }
      global_bin += getBin(coords[Ndim - 1], Ndim - 1);
      return global_bin;
    }

    ALPAKA_FN_ACC inline constexpr int getGlobalBinByBin(const VecArray<int32_t, Ndim>& Bins) const {
      int32_t globalBin = 0;
      for (auto dim = 0u; dim != Ndim; ++dim) {
        auto bin_i = wrapping[dim] ? (Bins[dim] % nperdim) : Bins[dim];
        globalBin += math::pow(static_cast<float>(nperdim), Ndim - dim - 1) * bin_i;
      }
      return globalBin;
    }

    ALPAKA_FN_ACC inline void searchBox(const SearchBoxExtremes<Ndim>& searchbox_extremes,
                                        SearchBoxBins<Ndim>& searchbox_bins) {
      for (auto dim = 0u; dim != Ndim; ++dim) {
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

    ALPAKA_FN_ACC inline constexpr float normalizeCoordinate(float coord, int dim) const {
      const float range = minmax->range(dim);
      float remainder = coord - static_cast<int>(coord / range) * range;
      if (remainder >= minmax->max(dim))
        remainder -= range;
      else if (remainder < minmax->min(dim))
        remainder += range;
      return remainder;
    }

    ALPAKA_FN_ACC inline auto distance(const std::array<float, Ndim + 1>& coord_i,
                                       const std::array<float, Ndim + 1>& coord_j) const {
      std::array<float, Ndim> distance_vector;
      for (auto dim = 0u; dim != Ndim; ++dim) {
        if (wrapping[dim])
          distance_vector[dim] = math::fabs(normalizeCoordinate(coord_i[dim] - coord_j[dim], dim));
        else
          distance_vector[dim] = math::fabs(coord_i[dim] - coord_j[dim]);
      }
      return distance_vector;
    }
  };

}  // namespace clue::internal
