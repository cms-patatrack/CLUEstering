
#pragma once

#include <array>
#include <type_traits>

namespace clue {
  namespace nostd {

    template <typename... Tn>
    inline constexpr auto make_array(Tn&&... args) {
      return std::array<std::common_type_t<Tn...>, sizeof...(Tn)>{{std::forward<Tn>(args)...}};
    }

  }  // namespace nostd
}  // namespace clue
