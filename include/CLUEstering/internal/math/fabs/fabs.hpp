#pragma once

#include <concepts>
#include <cmath>
#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/math/defines.hpp"

namespace clue::math {

  ALPAKA_FN_ACC inline constexpr float fabs(float arg) {
#if defined(CUDA_DEVICE_FN)
    return ::fabsf(arg);
#elif defined(HIP_DEVICE_FN)
    return ::fabsf(arg);
#elif defined(SYCL_DEVICE_FN)
    return sycl::fabs(arg);
#else
    // standard C/C++ code
    return ::fabsf(arg);
#endif
  }

  ALPAKA_FN_ACC inline constexpr double fabs(double arg) {
#if defined(CUDA_DEVICE_FN)
    return ::fabs(arg);
#elif defined(HIP_DEVICE_FN)
    return ::fabs(arg);
#elif defined(SYCL_DEVICE_FN)
    return sycl::fabs(arg);
#else
    // standard C/C++ code
    return ::fabs(arg);
#endif
  }

}  // namespace clue::math
