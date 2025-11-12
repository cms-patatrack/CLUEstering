/// @file AssociationMapView.hpp
/// @brief Definition of the AssociationMapView class, which provides a view into an association map data structure that
/// can be passed to kernels.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <concepts::device TDev>
  class AssociationMap;

  /// @brief A view into an association map data structure that can be passed to kernels.
  /// The AssociationMapView provides access to the underlying data of an AssociationMap without owning it.
  /// It allows for efficient retrieval of associated values for given keys, making it suitable for use inside kernels.
  class AssociationMapView {
  public:
    struct Extents {
      std::size_t keys;
      std::size_t values;
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
    /// @brief Get the extents of the association map.
    /// @return An Extents struct containing the number of values and keys in the association map.
    ALPAKA_FN_ACC auto extents() const { return m_extents; }

    /// @brief Get the associated values for a given key.
    ///
    /// @param key The key for which to get the associated values.
    /// @return A span containing the values associated with the given key.
    ALPAKA_FN_ACC auto operator[](size_t key) {
      auto size = m_offsets[key + 1] - m_offsets[key];
      auto* buf_ptr = m_indexes + m_offsets[key];
      return std::span<int32_t>{buf_ptr, static_cast<std::size_t>(size)};
    }
    /// @brief Get the associated values for a given key.
    ///
    /// @param key The key for which to get the associated values.
    /// @return A span containing the values associated with the given key.
    ALPAKA_FN_ACC auto operator[](size_t key) const {
      auto size = m_offsets[key + 1] - m_offsets[key];
      auto* buf_ptr = m_indexes + m_offsets[key];
      return std::span<const int32_t>{buf_ptr, static_cast<std::size_t>(size)};
    }
    /// @brief Get the number of associated values for a given key.
    ///
    /// @param key The key for which to get the count of associated values.
    /// @return The number of values associated with the given key.
    ALPAKA_FN_ACC auto count(std::size_t key) const { return m_offsets[key + 1] - m_offsets[key]; }
    /// @brief Check if there are any associated values for a given key.
    ///
    /// @param key The key to check for associated values.
    /// @return True if there are associated values for the given key, false otherwise.
    ALPAKA_FN_ACC bool contains(std::size_t key) const {
      return m_offsets[key + 1] > m_offsets[key];
    }
  };

}  // namespace clue
