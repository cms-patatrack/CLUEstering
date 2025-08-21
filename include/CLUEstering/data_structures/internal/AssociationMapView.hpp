
#pragma once

#include "CLUEstering/data_structures/internal/Span.hpp"

namespace clue {

  struct AssociationMapView {
    int32_t* m_indexes;
    int32_t* m_offsets;
    int32_t m_nelements;
    int32_t m_nbins;

    ALPAKA_FN_ACC Span<int32_t> indexes(size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return Span<int32_t>{buf_ptr, size};
    }
    ALPAKA_FN_ACC int32_t offsets(size_t bin_id) { return m_offsets[bin_id]; }
    ALPAKA_FN_ACC Span<int32_t> operator[](size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return Span<int32_t>{buf_ptr, size};
    }
  };

}  // namespace clue
