
#pragma once

#include <concepts>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue {
  namespace internal {
    namespace math {

      ALPAKA_FN_ACC inline constexpr float sqrt(float x) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::sqrt(x);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::sqrt(x);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::sqrt(x);
#else
        // standard C++ code
        return std::sqrt(x);
#endif
      }

      ALPAKA_FN_ACC inline constexpr double sqrt(double x) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::sqrt(x);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::sqrt(x);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::sqrt(x);
#else
        // standard C++ code
        return std::sqrt(x);
#endif
      }

      ALPAKA_FN_ACC inline constexpr float sqrtf(float x) { return sqrt(x); }

      template <std::integral T>
      ALPAKA_FN_ACC inline constexpr double sqrt(T x) {
        return sqrt(static_cast<double>(x));
      }

    }  // namespace math
  }  // namespace internal
}  // namespace clue
