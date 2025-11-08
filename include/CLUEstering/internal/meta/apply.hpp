
#pragma once

#include <type_traits>
#include <utility>

namespace clue::meta {

  template <std::size_t N, typename F>
    requires std::is_invocable_r_v<void, F()>
  void apply(F&& f) {
    [&]<std::size_t... Ids>(std::index_sequence<Ids...>) -> void {
      (f.template operator()<Ids>(), ...);
    }(std::make_index_sequence<N>{});
  }

}  // namespace clue::meta
