
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"

CLUE_NAMESPACE_BEGIN
namespace nostd {

  template <typename T>
  struct minimum {
    ALPAKA_FN_HOST_ACC constexpr T operator()(const T& a, const T& b) const {
      return std::min(a, b);
    }
  };

}  // namespace nostd
CLUE_NAMESPACE_END
