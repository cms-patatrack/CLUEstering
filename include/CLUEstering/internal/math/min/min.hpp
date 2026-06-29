
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/math/defines.hpp"
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <algorithm>
#endif

#if __STDCPP_FLOAT16_T__ == 1
#include <stdfloat>
#endif

namespace clue::math {

  template <clue::concepts::Numeric T>
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline T min(const T& a, const T& b) {
#if defined(CUDA_DEVICE_FN)
    return ::min(a, b);
#elif defined(HIP_DEVICE_FN)
    return ::min(a, b);
#elif defined(SYCL_DEVICE_FN)
    return sycl::min(a, b);
#else
    return std::min(a, b);
#endif
  }

  template <clue::concepts::Numeric T, typename Compare>
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline T min(const T& a, const T& b, Compare comp) {
#if defined(CUDA_DEVICE_FN)
    return ::min(a, b, comp);
#elif defined(HIP_DEVICE_FN)
    return ::min(a, b, comp);
#elif defined(SYCL_DEVICE_FN)
    return sycl::min(a, b, comp);
#else
    return std::min(a, b, comp);
#endif
  }

#if __STDCPP_FLOAT16_T__ == 1
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline std::float16_t min(const std::float16_t& a,
                                                            const std::float16_t& b) {
    const auto x = static_cast<float>(a);
    const auto y = static_cast<float>(b);
#if defined(CUDA_DEVICE_FN)
    return static_cast<std::float16_t>(::min(x, y));
#elif defined(HIP_DEVICE_FN)
    return static_cast<std::float16_t>(::min(x, y));
#elif defined(SYCL_DEVICE_FN)
    return static_cast<std::float16_t>(sycl::min(x, y));
#else
    return static_cast<std::float16_t>(std::min(x, y));
#endif
  }
#endif

}  // namespace clue::math
