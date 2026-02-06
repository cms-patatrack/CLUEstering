
#pragma once

#if defined(__CUDA_ARCH__)
#define CUDA_DEVICE_FN
#elif defined(__HIP_DEVICE_COMPILE__)
#define HIP_DEVICE_FN
#elif defined(__SYCL_DEVICE_ONLY__)
#define SYCL_DEVICE_FN
#endif

#if defined(_MSC_VER)
#define MATH_FN_CONSTEXPR
#else
#define MATH_FN_CONSTEXPR constexpr
#endif
