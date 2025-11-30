
#pragma once

#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace clue::meta {

  template <std::size_t N,
            typename F,
            typename Return = decltype(std::declval<F>().template operator()<0>())>
    requires std::is_arithmetic_v<Return>
  ALPAKA_FN_HOST_ACC inline constexpr auto accumulate(F&& f) {
    return [&]<std::size_t... Ids>(std::index_sequence<Ids...>) -> Return {
      return ((f.template operator()<Ids>()) + ... + Return{});
    }(std::make_index_sequence<N>{});
  }

}  // namespace clue::meta
