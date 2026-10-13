
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"

CLUE_NAMESPACE_BEGIN
namespace nostd {

  template <typename T>
  struct maximum {
    ALPAKA_FN_HOST_ACC constexpr T operator()(const T& a, const T& b) const {
      return std::max(a, b);
    }
  };

}  // namespace nostd
CLUE_NAMESPACE_END
