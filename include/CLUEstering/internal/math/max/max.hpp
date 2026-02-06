
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

}  // namespace clue::math
