
#pragma once

namespace clue::nostd {

  template <typename T>
  struct minimum {
    ALPAKA_FN_HOST_ACC constexpr T operator()(const T& a, const T& b) const {
      return std::min(a, b);
    }
  };

}  // namespace clue::nostd
