#pragma once

#include <concepts>
#include <cmath>
#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/math/defines.hpp"

namespace clue::math {

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float fabs(float arg) {
#if defined(CUDA_DEVICE_FN)
    return ::fabsf(arg);
#elif defined(HIP_DEVICE_FN)
    return ::fabsf(arg);
#elif defined(SYCL_DEVICE_FN)
    return sycl::fabs(arg);
#else
    return ::fabsf(arg);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double fabs(double arg) {
#if defined(CUDA_DEVICE_FN)
    return ::fabs(arg);
#elif defined(HIP_DEVICE_FN)
    return ::fabs(arg);
#elif defined(SYCL_DEVICE_FN)
    return sycl::fabs(arg);
#else
    return ::fabs(arg);
#endif
  }

}  // namespace clue::math
