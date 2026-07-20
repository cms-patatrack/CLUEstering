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

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double tgamma(double x) {
#if defined(CUDA_DEVICE_FN)
    return ::tgamma(x);
#elif defined(HIP_DEVICE_FN)
    return ::tgamma(x);
#else
    return std::tgamma(x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float tgamma(float x) {
#if defined(CUDA_DEVICE_FN)
    return ::tgammaf(x);
#elif defined(HIP_DEVICE_FN)
    return ::tgammaf(x);
#else
    return std::tgamma(x);
#endif
  }

}  // namespace clue::math