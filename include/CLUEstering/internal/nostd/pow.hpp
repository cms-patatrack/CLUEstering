
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>

CLUE_NAMESPACE_BEGIN
namespace nostd {

  template <std::integral T>
  constexpr T pow(T base, std::size_t exp) {
    T result = 1;
    for (auto i = 0u; i < exp; ++i) {
      result *= base;
    }
    return result;
  }

}  // namespace nostd
CLUE_NAMESPACE_END
