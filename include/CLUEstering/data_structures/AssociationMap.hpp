/// @file AssociationMap.hpp
/// @brief Provides the AssociationMap class for managing associations between keys and values
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/internal/Span.hpp"
#include "CLUEstering/data_structures/AssociationMapView.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <span>
#include <alpaka/alpaka.hpp>
#include <span>

namespace clue {

  template <concepts::device TDev>
  class AssociationMap;

  namespace internal {

    template <std::size_t Ndim, clue::concepts::device TDev>
    class Tiles;

    template <clue::concepts::queue TQueue>
    auto make_associator(TQueue& queue, std::span<const int32_t> associations, int32_t elements);
    auto make_associator(std::span<const int32_t> associations, int32_t elements)
        -> AssociationMap<alpaka::DevCpu>;
  }  // namespace internal

  /// @brief The AssociationMap class is a data structure that maps keys to values.
  /// It associates integer keys with integer values in ono-to-many or many-to-many associations.
  ///
  /// @tparam TDev The device type to use for the allocation. Defaults to clue::Device.
  template <concepts::device TDev = clue::Device>
  class AssociationMap {
  public:
    using key_type = int32_t;
    using mapped_type = int32_t;
    using value_type = std::pair<key_type, mapped_type>;
    using size_type = std::size_t;
    using iterator = mapped_type*;
    using const_iterator = const mapped_type*;
    using keys_container_type = device_buffer<TDev, key_type[]>;
    using mapped_container_type = device_buffer<TDev, mapped_type[]>;

    struct Extents {
      size_type keys;
      size_type values;
    };

    struct Containers {
      const keys_container_type& keys;
      const mapped_container_type& values;
    };

    /// @brief Construct an empty AssociationMap
    AssociationMap() = default;
    /// @brief Construct an AssociationMap with a specific number of elements and bins
    /// @param nelements The number of elements to allocate
    /// @param nbins The number of bins to allocate
    /// @note This constructor is only available for the CPU device
    AssociationMap(size_type nelements, size_type nbins)
      requires std::same_as<TDev, alpaka::DevCpu>;
    /// @brief Construct an AssociationMap with a specific number of elements and bins
    ///
    /// @param nelements The number of elements to allocate
    /// @param nbins The number of bins to allocate
    /// @param dev The device to use for the allocation
    AssociationMap(size_type nelements, size_type nbins, const TDev& dev);

    /// @brief Construct an AssociationMap with a specific number of elements and bins
    ///
    /// @param nelements The number of elements to allocate
    /// @param nbins The number of bins to allocate
    /// @param queue The queue to use for the allocation
    template <concepts::queue TQueue>
    AssociationMap(size_type nelements, size_type nbins, TQueue& queue);

    /// @brief Return the number of bins in the map
    ///
    /// @return The number of bins in the map
    auto size() const;
    /// @brief Return the extents of the internal buffers
    ///
    /// @return A struct containing the number of keys and values in the map
    auto extents() const;

    /// @brief Return iterator to the beginning of the content buffer
    /// @return An iterator to the beginning of the content buffer
    iterator begin();
    /// @brief Return const iterator to the beginning of the content buffer
    /// @return A const iterator to the beginning of the content buffer
    const_iterator begin() const;
    /// @brief Return const iterator to the beginning of the content buffer
    /// @return A const iterator to the beginning of the content buffer
    const_iterator cbegin() const;

    /// @brief Return iterator to the end of the content buffer
    /// @return An iterator to the end of the content buffer
    iterator end();
    /// @brief Return const iterator to the end of the content buffer
    /// @return A const iterator to the end of the content buffer
    const_iterator end() const;
    /// @brief Return const iterator to the end of the content buffer
    /// @return A const iterator to the end of the content buffer
    const_iterator cend() const;

    // TODO: the STL implementation for std::flat_multimap returns any element with the given key,
    // Should we do the same? Should we return the first element or return a pair that gives the entire range?
    // In the first case it would be equavalent to lower_bound, in the second case it would be equivalent to equal_range.
    iterator find(key_type key);
    const_iterator find(key_type key) const;

    /// @brief Count the number of elements with the given key
    ///
    /// @param key The key to count
    /// @return The number of elements associated to a given key
    size_type count(key_type key) const;

    /// @brief Check if the map contains elements with a given key
    ///
    /// @param key The key to check
    /// @return True if the map contains elements with the given key, false otherwise
    bool contains(key_type key) const;

    /// @brief Get the iterator to the first element with a given key
    ///
    /// @param key The key to search for
    /// @return An iterator to the first element with the given key
    iterator lower_bound(key_type key);
    /// @brief Get the const iterator to the first element with a given key
    ///
    /// @param key The key to search for
    /// @return A const iterator to the first element with the given key
    const_iterator lower_bound(key_type key) const;

    /// @brief Get the iterator to the first element with a key greater than the given key
    ///
    /// @param key The key to search for
    /// @return An iterator to the first element with a key greater than the given key
    iterator upper_bound(key_type key);
    /// @brief Get the const iterator to the first element with a key greater than the given key
    ///
    /// @param key The key to search for
    /// @return A const iterator to the first element with a key greater than the given key
    const_iterator upper_bound(key_type key) const;

    /// @brief Get the range of elements with a given key
    ///
    /// @param key The key to search for
    /// @return A pair of iterators representing the range of elements with the given key
    std::pair<iterator, iterator> equal_range(key_type key);
    /// @brief Get the const range of elements with a given key
    ///
    /// @param key The key to search for
    /// @return A pair of const iterators representing the range of elements with the given key
    std::pair<const_iterator, const_iterator> equal_range(key_type key) const;

    /// @brief Get a const-reference to the underlying containers
    ///
    /// @return A Containers object
    Containers extract() const;

    /// @brief Get the constant view of the association map
    /// @return A const reference to the AssociationMapView
    const AssociationMapView& view() const;
    /// @brief Get the view of the association map
    /// @return A reference to the AssociationMapView
    AssociationMapView& view();

  private:
    device_buffer<TDev, mapped_type[]> m_indexes;
    device_buffer<TDev, key_type[]> m_offsets;
    AssociationMapView m_view;
    Extents m_extents;

    ALPAKA_FN_HOST void initialize(size_type nelements, size_type nbins)
      requires std::same_as<TDev, alpaka::DevCpu>;
    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(size_type nelements, size_type nbins, TQueue& queue);

    ALPAKA_FN_HOST void reset(size_type nelements, size_type nbins);

    template <concepts::accelerator TAcc, typename TFunc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, TFunc func, TQueue& queue);
    ALPAKA_FN_HOST void fill(std::span<const key_type> associations)
      requires std::same_as<TDev, alpaka::DevCpu>;
    template <concepts::accelerator TAcc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, std::span<const key_type> associations, TQueue& queue);

    ALPAKA_FN_HOST const auto& indexes() const;
    ALPAKA_FN_HOST auto& indexes();

    ALPAKA_FN_HOST const device_buffer<TDev, int32_t[]>& offsets() const;
    ALPAKA_FN_HOST device_buffer<TDev, int32_t[]>& offsets();

    template <concepts::device _TDev>
    friend class Followers;

    template <std::size_t Ndim, concepts::device _TDev>
    friend class internal::Tiles;

    template <concepts::queue _TQueue>
    friend auto clue::internal::make_associator(_TQueue&, std::span<const int32_t>, int32_t);
    friend auto clue::internal::make_associator(std::span<const int32_t>, int32_t)
        -> AssociationMap<alpaka::DevCpu>;
  };

  using host_associator = AssociationMap<alpaka::DevCpu>;

}  // namespace clue

#include "CLUEstering/data_structures/detail/AssociationMap.hpp"
