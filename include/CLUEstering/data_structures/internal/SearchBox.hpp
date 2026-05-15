
#pragma once

#include <array>
#include <cassert>
#include <cstdint>

namespace clue {

  template <std::size_t Ndim, typename T>
  class SearchBox {
  public:
    constexpr auto& operator[](int dim) {
      assert(dim >= 0 && static_cast<std::size_t>(dim) < Ndim);
      return m_extremes[dim];
    }
    constexpr const auto& operator[](int dim) const {
      assert(dim >= 0 && static_cast<std::size_t>(dim) < Ndim);
      return m_extremes[dim];
    }

    constexpr auto size() const { return Ndim; }

  private:
    std::array<std::array<T, 2>, Ndim> m_extremes;
  };

  template <std::size_t Ndim, std::floating_point TData = float>
  using SearchBoxExtremes = SearchBox<Ndim, std::remove_cv_t<std::remove_reference_t<TData>>>;

  template <std::size_t Ndim>
  using SearchBoxBins = SearchBox<Ndim, int32_t>;

}  // namespace clue
