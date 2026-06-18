
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace clue::nostd {

  template <std::integral T>
  constexpr T pow(T base, std::size_t exp) {
    T result = 1;
    for (auto i = 0u; i < exp; ++i) {
      result *= base;
    }
    return result;
  }

}  // namespace clue::nostd
