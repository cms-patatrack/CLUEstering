
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <concepts::device TDev>
  class AssociationMap;

  class AssociationMapView {
  public:
    struct Extents {
      std::size_t values;
      std::size_t keys;
    };

  private:
    int32_t* m_indexes;
    int32_t* m_offsets;
    Extents m_extents;

    AssociationMapView() = default;
    AssociationMapView(int32_t* indexes, int32_t* offsets, std::size_t nvalues, std::size_t nkeys)
        : m_indexes(indexes), m_offsets(offsets), m_extents{nvalues, nkeys} {}

    template <concepts::device TDev>
    friend class AssociationMap;

  public:
    ALPAKA_FN_ACC auto extents() const { return m_extents; }

    ALPAKA_FN_ACC auto indexes(std::size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return std::span<int32_t>{buf_ptr, static_cast<std::size_t>(size)};
    }
    ALPAKA_FN_ACC auto indexes(std::size_t bin_id) const {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return std::span<const int32_t>{buf_ptr, static_cast<std::size_t>(size)};
    }

    ALPAKA_FN_ACC auto operator[](size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return std::span<int32_t>{buf_ptr, static_cast<std::size_t>(size)};
    }
    ALPAKA_FN_ACC auto operator[](size_t bin_id) const {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return std::span<const int32_t>{buf_ptr, static_cast<std::size_t>(size)};
    }
    ALPAKA_FN_ACC auto count(std::size_t key) const { return m_offsets[key + 1] - m_offsets[key]; }
    ALPAKA_FN_ACC bool contains(std::size_t key) const {
      return m_offsets[key + 1] > m_offsets[key];
    }
  };

}  // namespace clue
