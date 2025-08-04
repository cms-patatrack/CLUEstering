
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue {
  namespace internal {
    namespace math {

      ALPAKA_FN_ACC inline constexpr float sqrt(float x) {
#if defined(CUDA_DEVICE_FN)
        // CUDA device code
        return ::sqrt(x);
#elif defined(HIP_DEVICE_FN)
        // HIP/ROCm device code
        return ::sqrt(x);
#elif defined(SYCL_DEVICE_FN)
        // SYCL device code
        return sycl::sqrt(x);
#else
        // standard C++ code
        return std::sqrt(x);
#endif
      }

      ALPAKA_FN_ACC inline constexpr double sqrt(double x) {
#if defined(CUDA_DEVICE_FN)
        // CUDA device code
        return ::sqrt(x);
#elif defined(HIP_DEVICE_FN)
        // HIP/ROCm device code
        return ::sqrt(x);
#elif defined(SYCL_DEVICE_FN)
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
