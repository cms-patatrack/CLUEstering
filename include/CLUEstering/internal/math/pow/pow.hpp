
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

#if __STDCPP_FLOAT16_T__ == 1
#include <stdfloat>
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

#if __STDCPP_FLOAT16_T__ == 1
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline std::float16_t pow(std::float16_t base,
                                                            std::float16_t exp) {
    const auto b = static_cast<float>(base);
    const auto e = static_cast<float>(exp);
#if defined(CUDA_DEVICE_FN)
    return static_cast<std::float16_t>(::pow(b, e));
#elif defined(HIP_DEVICE_FN)
    return static_cast<std::float16_t>(::pow(b, e));
#elif defined(SYCL_DEVICE_FN)
    return static_cast<std::float16_t>(sycl::pow(b, e));
#else
    return static_cast<std::float16_t>(std::pow(b, e));
#endif
  }
#endif

}  // namespace clue::math
