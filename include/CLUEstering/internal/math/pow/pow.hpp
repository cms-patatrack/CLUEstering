
#pragma once

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue {
  namespace internal {
    namespace math {

      ALPAKA_FN_ACC inline constexpr float pow(float base, float exp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::pow(base, exp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::pow(base, exp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::pow(base, exp);
#else
        // standard C++ code
        return std::pow(base, exp);
#endif
      }

      ALPAKA_FN_ACC inline constexpr double pow(double base, double exp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // CUDA device code
        return ::pow(base, exp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // HIP/ROCm device code
        return ::pow(base, exp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        // SYCL device code
        return sycl::pow(base, exp);
#else
        // standard C++ code
        return std::pow(base, exp);
#endif
      }

      ALPAKA_FN_ACC inline constexpr float powf(float base, float exp) { return pow(base, exp); }

    }  // namespace math
  }  // namespace internal
}  // namespace clue
