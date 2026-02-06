
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue::math {

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float sqrt(float x) {
#if defined(CUDA_DEVICE_FN)
    return ::sqrt(x);
#elif defined(HIP_DEVICE_FN)
    return ::sqrt(x);
#elif defined(SYCL_DEVICE_FN)
    return sycl::sqrt(x);
#else
    return std::sqrt(x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double sqrt(double x) {
#if defined(CUDA_DEVICE_FN)
    return ::sqrt(x);
#elif defined(HIP_DEVICE_FN)
    return ::sqrt(x);
#elif defined(SYCL_DEVICE_FN)
    return sycl::sqrt(x);
#else
    return std::sqrt(x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float sqrtf(float x) { return sqrt(x); }

  template <std::integral T>
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double sqrt(T x) {
    return sqrt(static_cast<double>(x));
  }

}  // namespace clue::math
