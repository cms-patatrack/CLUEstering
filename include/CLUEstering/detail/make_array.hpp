
#pragma once

#include <array>
#include <type_traits>

namespace clue {
  namespace nostd {

    template <typename... Tn>
    inline constexpr auto make_array(Tn&&... args) {
      return std::array<std::common_type_t<Tn...>, sizeof...(Tn)>{{std::forward<Tn>(args)...}};
    }

    template <typename T, std::size_t N>
    inline constexpr auto make_array(T value) {
      std::array<T, N> arr;
      [=, &arr]<std::size_t... Ids>(std::index_sequence<Ids...>) -> void {
        ((arr[Ids] = value), ...);
      }(std::make_index_sequence<N>{});
      return arr;
    }

  }  // namespace nostd
}  // namespace clue
