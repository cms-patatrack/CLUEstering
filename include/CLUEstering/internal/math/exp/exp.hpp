
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

#if __STDCPP_FLOAT16_T__ == 1
#include <stdfloat>
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

#if __STDCPP_FLOAT16_T__ == 1
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline std::float16_t exp(std::float16_t x) {
    const auto y = static_cast<float>(x);
#if defined(CUDA_DEVICE_FN)
    return static_cast<std::float16_t>(::exp(y));
#elif defined(HIP_DEVICE_FN)
    return static_cast<std::float16_t>(::exp(y));
#elif defined(SYCL_DEVICE_FN)
    return static_cast<std::float16_t>(sycl::exp(y));
#else
    return static_cast<std::float16_t>(std::exp(y));
#endif
  }
#endif

}  // namespace clue::math
