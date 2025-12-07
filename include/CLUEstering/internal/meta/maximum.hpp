
#pragma once

#include <algorithm>
#include <cstddef>

namespace clue::meta {

  template <std::size_t N,
            typename F,
            typename Return = decltype(std::declval<F>().template operator()<0>())>
    requires(N >= 1)
  ALPAKA_FN_HOST_ACC constexpr inline auto maximum(F&& f) {
    Return max = Return{};
    [&]<std::size_t... Dims>(std::index_sequence<Dims...>) {
      ((max = std::max(max, f.template operator()<Dims>())), ...);
    }(std::make_index_sequence<N>{});
    return max;
  }

}  // namespace clue::meta
