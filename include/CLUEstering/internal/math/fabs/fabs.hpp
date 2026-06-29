#pragma once

#include <concepts>
#include <cmath>
#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/math/defines.hpp"

#if __STDCPP_FLOAT16_T__ == 1
#include <stdfloat>
#endif

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


#if __STDCPP_FLOAT16_T__ == 1
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float fabs(std::float16_t arg) {
    const auto y = static_cast<float>(arg);
#if defined(CUDA_DEVICE_FN)
    return static_cast<std::float16_t>(::fabsf(y));
#elif defined(HIP_DEVICE_FN)
    return static_cast<std::float16_t>(::fabsf(y));
#elif defined(SYCL_DEVICE_FN)
    return static_cast<std::float16_t>(sycl::fabs(y));
#else
    return static_cast<std::float16_t>(::fabsf(y));
#endif
  }
#endif

}  // namespace clue::math
