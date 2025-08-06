
#pragma once

#include "CLUEstering/data_structures/internal/Span.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <alpaka/alpaka.hpp>

namespace clue {

  namespace concepts = detail::concepts;

  struct AssociationMapView;

  template <concepts::device TDev>
  class AssociationMap {
  public:
    using key_type = int32_t;
    using mapped_type = int32_t;
    using value_type = std::pair<key_type, mapped_type>;
    using size_type = std::size_t;
    using iterator = mapped_type*;
    using const_iterator = const mapped_type*;

    struct Extents {
      size_type keys;
      size_type values;
    };

    AssociationMap() = default;
    AssociationMap(size_type nelements, size_type nbins, const TDev& dev);

    template <concepts::queue TQueue>
    AssociationMap(size_type nelements, size_type nbins, TQueue& queue);

    auto size() const;
    auto extents() const;

    iterator begin();
    const_iterator begin() const;
    const_iterator cbegin() const;

    iterator end();
    const_iterator end() const;
    const_iterator cend() const;

    // TODO: the STL implementation for std::flat_multimap returns any element with the given key,
    // Should we do the same? Should we return the first element or return a pair that gives the entire range?
    // In the first case it would be equavalent to lower_bound, in the second case it would be equivalent to equal_range.
    iterator find(key_type key);
    const_iterator find(key_type key) const;

    size_type count(key_type key) const;

    bool contains(key_type key) const;

    iterator lower_bound(key_type key);
    const_iterator lower_bound(key_type key) const;

    iterator upper_bound(key_type key);
    const_iterator upper_bound(key_type key) const;

    std::pair<iterator, iterator> equal_range(key_type key);
    std::pair<const_iterator, const_iterator> equal_range(key_type key) const;

  private:
    device_buffer<TDev, mapped_type[]> m_indexes;
    device_buffer<TDev, key_type[]> m_offsets;
    host_buffer<AssociationMapView> m_hview;
    device_buffer<TDev, AssociationMapView> m_view;
    size_type m_nbins;

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(size_type nelements, size_type nbins, TQueue& queue);

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void reset(TQueue& queue, size_type nelements, size_type nbins);

    template <concepts::accelerator TAcc, typename TFunc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, TFunc func, TQueue& queue);

    template <concepts::accelerator TAcc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, std::span<key_type> associations, TQueue& queue);

    AssociationMapView* view();

    ALPAKA_FN_HOST const auto& indexes() const;
    ALPAKA_FN_HOST auto& indexes();

    ALPAKA_FN_ACC Span<int32_t> indexes(size_type bin_id);
    ALPAKA_FN_HOST device_view<TDev, int32_t[]> indexes(const TDev& dev, size_type bin_id);
    ALPAKA_FN_ACC Span<int32_t> operator[](size_type bin_id);

    ALPAKA_FN_HOST const device_buffer<TDev, int32_t[]>& offsets() const;
    ALPAKA_FN_HOST device_buffer<TDev, int32_t[]>& offsets();

    ALPAKA_FN_ACC int32_t offsets(size_type bin_id) const;

    template <concepts::device _TDev>
    friend class Followers;

    template <uint8_t Ndim, concepts::device _TDev>
    friend class TilesAlpaka;
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/AssociationMap.hpp"
