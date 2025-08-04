
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

      ALPAKA_FN_ACC inline constexpr float exp(float x) {
#if defined(CUDA_DEVICE_FN)
        // CUDA device code
        return ::exp(x);
#elif defined(HIP_DEVICE_FN)
        // HIP/ROCm device code
        return ::exp(x);
#elif defined(SYCL_DEVICE_FN)
        // SYCL device code
        return sycl::exp(x);
#else
        // standard C++ code
        return std::exp(x);
#endif
      }

      ALPAKA_FN_ACC inline constexpr double exp(double x) {
#if defined(CUDA_DEVICE_FN)
        // CUDA device code
        return ::exp(x);
#elif defined(HIP_DEVICE_FN)
        // HIP/ROCm device code
        return ::exp(x);
#elif defined(SYCL_DEVICE_FN)
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
