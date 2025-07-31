
#pragma once

#include "CLUEstering/detail/concepts.hpp"

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue {
  namespace internal {
    namespace math {

      template <clue::detail::concepts::Numeric T>
      ALPAKA_FN_ACC inline constexpr const T& max(const T& a, const T& b) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::max(a, b);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::max(a, b);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::max(a, b);
#else
        // standard C++ code
        return std::max(a, b);
#endif
      }

      template <clue::detail::concepts::Numeric T, typename Compare>
      ALPAKA_FN_ACC inline constexpr const T& max(const T& a, const T& b, Compare comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::max(a, b, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::max(a, b, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::max(a, b, comp);
#else
        // standard C++ code
        return std::max(a, b, comp);
#endif
      }

    }  // namespace math
  }  // namespace internal
}  // namespace clue
