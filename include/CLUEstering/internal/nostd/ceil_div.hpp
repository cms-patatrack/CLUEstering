
#pragma once

#include <concepts>

namespace clue::nostd {

  template <std::integral T1, std::integral T2>
  constexpr auto ceil_div(T1 numerator, T2 denominator) -> T1 {
    return static_cast<T1>((numerator + denominator - 1) / denominator);
  }

}  // namespace clue::nostd
