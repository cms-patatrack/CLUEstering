
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


#ifdef __STDCPP_FLOAT16_T__
#include <stdfloat>

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float sqrt(std::float16_t x) {
    const auto y = static_cast<float>(x);
#if defined(CUDA_DEVICE_FN)
    return static_cast<std::float16_t>(::sqrt(y));
#elif defined(HIP_DEVICE_FN)
    return static_cast<std::float16_t>(::sqrt(y));
#elif defined(SYCL_DEVICE_FN)
    return static_cast<std::float16_t>(sycl::sqrt(y));
#else
    return static_cast<std::float16_t>(std::sqrt(y));
#endif
  }
#endif

}  // namespace clue::math
