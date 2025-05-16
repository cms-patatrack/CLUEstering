
#pragma once

#include <alpaka/alpaka.hpp>

namespace clue {
  namespace detail {
    namespace concepts {

      template <typename T>
      concept queue = alpaka::isQueue<T>;

      template <typename T>
      concept device = alpaka::isDevice<T>;

      template <typename T>
      concept accelerator = alpaka::isAccelerator<T>;

      template <typename T>
      concept platform = alpaka::isPlatform<T>;

    }  // namespace concepts
  }  // namespace detail
}  // namespace clue
