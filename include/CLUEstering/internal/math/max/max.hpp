
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/math/defines.hpp"
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue::math {

  template <clue::concepts::Numeric T>
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline T max(const T& a, const T& b) {
#if defined(CUDA_DEVICE_FN)
    return ::max(a, b);
#elif defined(HIP_DEVICE_FN)
    return ::max(a, b);
#elif defined(SYCL_DEVICE_FN)
    return sycl::max(a, b);
#else
    return std::max(a, b);
#endif
  }

  template <clue::concepts::Numeric T, typename Compare>
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline T max(const T& a, const T& b, Compare comp) {
#if defined(CUDA_DEVICE_FN)
    return ::max(a, b, comp);
#elif defined(HIP_DEVICE_FN)
    return ::max(a, b, comp);
#elif defined(SYCL_DEVICE_FN)
    return sycl::max(a, b, comp);
#else
    return std::max(a, b, comp);
#endif
  }


#ifdef __STDCPP_FLOAT16_T__
#include <stdfloat>

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float max(const std::float16_t& a, const std::float16_t& b) {
    const auto x = static_cast<float>(a);
    const auto y = static_cast<float>(b);
#if defined(CUDA_DEVICE_FN)
    return static_cast<std::float16_t>(::max(x, y));
#elif defined(HIP_DEVICE_FN)
    return static_cast<std::float16_t>(::max(x, y));
#elif defined(SYCL_DEVICE_FN)
    return static_cast<std::float16_t>(sycl::max(x, y));
#else
    return static_cast<std::float16_t>(std::max(x, y));
#endif
  }
#endif

}  // namespace clue::math
