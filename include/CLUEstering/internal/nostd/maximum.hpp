
#pragma once

namespace clue::nostd {

  template <typename T>
  struct maximum {
    ALPAKA_FN_HOST_ACC constexpr T operator()(const T& a, const T& b) const {
      return std::max(a, b);
    }
  };

}  // namespace clue::nostd
