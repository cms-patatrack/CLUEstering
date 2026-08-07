#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>
#include "../sqrt/sqrt.hpp"

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <cmath>
#endif
#include <cmath>



namespace clue::math {

  ALPAKA_FN_ACC inline double cyl_bessel_j(int n, double x) {
#if defined(CUDA_DEVICE_FN)
    return ::jn(n, x);
#elif defined(HIP_DEVICE_FN)
    return ::jn(n, x);
#else
    return ::std::cyl_bessel_j(n, x);
#endif
  }

     ALPAKA_FN_ACC inline float cyl_bessel_j(int n, float x) {
#if defined(CUDA_DEVICE_FN)
    return ::jnf(n, x);
#elif defined(HIP_DEVICE_FN)
    return ::jnf(n, x);
#else
    return ::std::cyl_bessel_jf(static_cast<float>(n), x);
#endif
  }


  template <std::size_t Ndim, std::floating_point TData>
float cyl_bessel_j(float x){
    if constexpr (Ndim == 1) {
        return sqrt(TData{2} / (TData{M_PI} * x)) * sin(x);  
    }
    else if constexpr (Ndim == 2) {
        return cyl_bessel_j(1, x);
    }
    else if constexpr (Ndim == 3) {
        return sqrt(TData{2} / (TData{M_PI} * x)) * ( (sin(x) / x) - cos(x));
    }
    else if constexpr (Ndim == 4) {
        return cyl_bessel_j(2, x);
    }
}

}  // namespace clue::math