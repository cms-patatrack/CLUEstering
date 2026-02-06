
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue::math {

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float pow(float base, float exp) {
#if defined(CUDA_DEVICE_FN)
    return ::pow(base, exp);
#elif defined(HIP_DEVICE_FN)
    return ::pow(base, exp);
#elif defined(SYCL_DEVICE_FN)
    return sycl::pow(base, exp);
#else
    return std::pow(base, exp);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double pow(double base, double exp) {
#if defined(CUDA_DEVICE_FN)
    return ::pow(base, exp);
#elif defined(HIP_DEVICE_FN)
    return ::pow(base, exp);
#elif defined(SYCL_DEVICE_FN)
    return sycl::pow(base, exp);
#else
    return std::pow(base, exp);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float powf(float base, float exp) {
    return pow(base, exp);
  }

}  // namespace clue::math
