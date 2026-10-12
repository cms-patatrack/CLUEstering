
#pragma once

#include <alpaka/alpaka.hpp>

namespace clue::concepts {

  template <typename T>
  concept queue = alpaka::concepts::Queue<T>;

  template <typename T>
  concept device = alpaka::concepts::Device<T>;

  template <typename T>
  concept accelerator = alpaka::concepts::Acc<T>;

  template <typename T>
  concept platform = alpaka::concepts::Platform<T>;

  template <typename T>
  concept pointer = std::is_pointer_v<T>;

  template <typename T>
  concept Numeric = requires {
    std::is_arithmetic_v<T>;
    requires sizeof(T) <= 8;
  };

}  // namespace clue::concepts
