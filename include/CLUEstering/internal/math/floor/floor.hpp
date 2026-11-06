#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include <concepts>
#include <cmath>
#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/math/defines.hpp"

#if __STDCPP_FLOAT16_T__ == 1
#include <stdfloat>
#endif

CLUE_NAMESPACE_BEGIN
namespace math {

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float floor(float arg) {
#if defined(CUDA_DEVICE_FN)
    return ::floorf(arg);
#elif defined(HIP_DEVICE_FN)
    return ::floorf(arg);
#elif defined(SYCL_DEVICE_FN)
    return sycl::floor(arg);
#else
    return ::floorf(arg);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double floor(double arg) {
#if defined(CUDA_DEVICE_FN)
    return ::floor(arg);
#elif defined(HIP_DEVICE_FN)
    return ::floor(arg);
#elif defined(SYCL_DEVICE_FN)
    return sycl::floor(arg);
#else
    return ::floor(arg);
#endif
  }

#if __STDCPP_FLOAT16_T__ == 1
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline std::float16_t floor(std::float16_t arg) {
    const auto y = static_cast<float>(arg);
#if defined(CUDA_DEVICE_FN)
    return static_cast<std::float16_t>(::floorf(y));
#elif defined(HIP_DEVICE_FN)
    return static_cast<std::float16_t>(::floorf(y));
#elif defined(SYCL_DEVICE_FN)
    return static_cast<std::float16_t>(sycl::floor(y));
#else
    return static_cast<std::float16_t>(::floorf(y));
#endif
  }
#endif

}  // namespace math
CLUE_NAMESPACE_END
