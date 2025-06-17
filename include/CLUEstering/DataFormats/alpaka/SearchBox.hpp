
#pragma once

#include <array>
#include <cstdint>

namespace clue {

  template <uint8_t Ndim, typename T>
  class SearchBox {
  public:
    constexpr auto& operator[](int dim) { return m_extremes[dim]; }
    constexpr const auto& operator[](int dim) const { return m_extremes[dim]; }

    constexpr auto size() const { return Ndim; }

  private:
    std::array<std::array<T, 2>, Ndim> m_extremes;
  };

  template <uint8_t Ndim>
  using SearchBoxExtremes = SearchBox<Ndim, float>;

  template <uint8_t Ndim>
  using SearchBoxBins = SearchBox<Ndim, int32_t>;

}  // namespace clue
