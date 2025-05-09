
#pragma once

#include <array>
#include <cstdint>

namespace clue {

  template <uint8_t Ndim, typename T>
  class SearchBox {
  public:
    auto& operator[](int dim) { return m_extremes[dim]; }
    const auto& operator[](int dim) const { return m_extremes[dim]; }

  private:
    std::array<std::array<T, 2>, Ndim> m_extremes;
  };

  template <uint8_t Ndim>
  using SearchBoxExtremes = SearchBox<Ndim, float>;

  template <uint8_t Ndim>
  using SearchBoxBins = SearchBox<Ndim, uint32_t>;

}  // namespace clue
