
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue {
  namespace internal {
    namespace math {

      ALPAKA_FN_ACC inline constexpr float pow(float base, float exp) {
#if defined(CUDA_DEVICE_FN)
        // CUDA device code
        return ::pow(base, exp);
#elif defined(HIP_DEVICE_FN)
        // HIP/ROCm device code
        return ::pow(base, exp);
#elif defined(SYCL_DEVICE_FN)
        // SYCL device code
        return sycl::pow(base, exp);
#else
        // standard C++ code
        return std::pow(base, exp);
#endif
      }

      ALPAKA_FN_ACC inline constexpr double pow(double base, double exp) {
#if defined(CUDA_DEVICE_FN)
        // CUDA device code
        return ::pow(base, exp);
#elif defined(HIP_DEVICE_FN)
        // HIP/ROCm device code
        return ::pow(base, exp);
#elif defined(SYCL_DEVICE_FN)
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
