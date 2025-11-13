
#pragma once

#include "CLUEstering/internal/meta/apply.hpp"
#include <array>
#include <type_traits>

namespace clue::nostd {

  template <typename... Tn>
  inline constexpr auto make_array(Tn&&... args) {
    return std::array<std::common_type_t<Tn...>, sizeof...(Tn)>{{std::forward<Tn>(args)...}};
  }

  template <typename T, std::size_t N>
  inline constexpr auto make_array(T value) {
    std::array<T, N> arr;
    meta::apply<N>([=, &arr]<std::size_t Idx>() { arr[Idx] = value; });
    return arr;
  }

}  // namespace clue::nostd
