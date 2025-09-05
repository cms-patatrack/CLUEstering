
#pragma once

#include "CLUEstering/core/detail/defines.hpp"
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

    template <typename TKernel>
    concept convolutional_kernel =
        requires(TKernel&& k, const internal::Acc& acc, float d, int i, int j) {
          { k(acc, d, i, j) } -> std::same_as<float>;
        };

  }  // namespace concepts
}  // namespace clue
