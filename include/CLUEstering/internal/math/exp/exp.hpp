
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue::math {

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float exp(float x) {
#if defined(CUDA_DEVICE_FN)
    return ::exp(x);
#elif defined(HIP_DEVICE_FN)
    return ::exp(x);
#elif defined(SYCL_DEVICE_FN)
    return sycl::exp(x);
#else
    return std::exp(x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double exp(double x) {
#if defined(CUDA_DEVICE_FN)
    return ::exp(x);
#elif defined(HIP_DEVICE_FN)
    return ::exp(x);
#elif defined(SYCL_DEVICE_FN)
    return sycl::exp(x);
#else
    return std::exp(x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float expf(float x) { return exp(x); }

  template <std::integral T>
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double exp(T x) {
    return exp(static_cast<double>(x));
  }

}  // namespace clue::math
