
#pragma once

#include <alpaka/alpaka.hpp>

namespace clue {
  namespace concepts {

    template <typename T>
    concept queue = alpaka::isQueue<T>;

    template <typename T>
    concept device = alpaka::isDevice<T>;

    template <typename T>
    concept accelerator = alpaka::isAccelerator<T>;

    template <typename T>
    concept platform = alpaka::isPlatform<T>;

    template <typename T>
    concept Numeric = requires {
      std::is_arithmetic_v<T>;
      requires sizeof(T) <= 8;
    };

  }  // namespace concepts
}  // namespace clue
