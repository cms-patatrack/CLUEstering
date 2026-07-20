#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <cmath>
#endif

#if __STDCPP_FLOAT16_T__ == 1
#include <stdfloat>
#endif

namespace clue::math {

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double cyl_bessel_j(int n, double x) {
#if defined(CUDA_DEVICE_FN)
    return ::jn(n, x);
#elif defined(HIP_DEVICE_FN)
    return ::jn(n, x);
#else
    return std::cyl_bessel_j(n, x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float cyl_bessel_jf(int n, float x) {
#if defined(CUDA_DEVICE_FN)
    return ::jnf(n, x);
#elif defined(HIP_DEVICE_FN)
    return ::jnf(n, x);
#else
    return std::cyl_bessel_j(n, x);
#endif
  }

}  // namespace clue::math