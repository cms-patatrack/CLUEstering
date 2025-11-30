
#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <alpaka/alpaka.hpp>

namespace clue::meta {

  template <std::size_t N, typename F>
    requires std::is_invocable_r_v<void, F()>
  ALPAKA_FN_HOST_ACC inline constexpr void apply(F&& f) {
    [&]<std::size_t... Ids>(std::index_sequence<Ids...>) -> void {
      (f.template operator()<Ids>(), ...);
    }(std::make_index_sequence<N>{});
  }

}  // namespace clue::meta
