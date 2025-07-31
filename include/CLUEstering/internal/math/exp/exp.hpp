
#pragma once

#include <concepts>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue {
  namespace internal {
    namespace math {

      ALPAKA_FN_ACC inline constexpr float exp(float x) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::exp(x);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::exp(x);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::exp(x);
#else
        // standard C++ code
        return std::exp(x);
#endif
      }

      ALPAKA_FN_ACC inline constexpr double exp(double x) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::exp(x);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::exp(x);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::exp(x);
#else
        // standard C++ code
        return std::exp(x);
#endif
      }

      ALPAKA_FN_ACC inline constexpr float expf(float x) { return exp(x); }

      template <std::integral T>
      ALPAKA_FN_ACC inline constexpr double exp(T x) {
        return exp(static_cast<double>(x));
      }

    }  // namespace math
  }  // namespace internal
}  // namespace clue
