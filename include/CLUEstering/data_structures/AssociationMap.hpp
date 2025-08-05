
#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <cassert>

#include <limits>

#include <alpaka/alpaka.hpp>

#include "CLUEstering/data_structures/internal/Span.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/algorithm/scan/scan.hpp"

namespace clue {

  namespace concepts = detail::concepts;

  struct AssociationMapView;

  template <concepts::device TDev>
  class AssociationMap {
  public:
    using key_type = int32_t;
    using mapped_type = int32_t;
    using size_type = std::size_t;

    struct Extents {
      size_type content;
      size_type offset;
    };

    AssociationMap() = default;
    AssociationMap(size_type nelements, size_type nbins, const TDev& dev);

    template <concepts::queue TQueue>
    AssociationMap(size_type nelements, size_type nbins, TQueue& queue);

    AssociationMapView* view();

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(size_type nelements, size_type nbins, TQueue& queue);

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void reset(TQueue& queue, size_type nelements, size_type nbins);

    auto size() const;

    auto extents() const;

    ALPAKA_FN_HOST const auto& indexes() const;
    ALPAKA_FN_HOST auto& indexes();

    ALPAKA_FN_ACC Span<int32_t> indexes(size_type bin_id);
    ALPAKA_FN_HOST device_view<TDev, int32_t[]> indexes(const TDev& dev, size_type bin_id);
    ALPAKA_FN_ACC Span<int32_t> operator[](size_type bin_id);

    ALPAKA_FN_HOST const device_buffer<TDev, int32_t[]>& offsets() const;
    ALPAKA_FN_HOST device_buffer<TDev, int32_t[]>& offsets();

    ALPAKA_FN_ACC int32_t offsets(size_type bin_id) const;

    template <concepts::accelerator TAcc, typename TFunc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, TFunc func, TQueue& queue);

    template <concepts::accelerator TAcc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, std::span<key_type> associations, TQueue& queue);

  private:
    device_buffer<TDev, mapped_type[]> m_indexes;
    device_buffer<TDev, key_type[]> m_offsets;
    host_buffer<AssociationMapView> m_hview;
    device_buffer<TDev, AssociationMapView> m_view;
    size_type m_nbins;
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/AssociationMap.hpp"
