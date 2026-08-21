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

    namespace detail {

        template <typename TData>
        ALPAKA_FN_ACC inline TData gamma(TData nu) {
         TData result = static_cast<TData>(1);
        // Gamma(nu+1)= nu * Gamma(nu)
        while (nu > static_cast<TData>(1)) {
            nu -= static_cast<TData>(1);
            result *= nu;
        }
        if (nu == static_cast<TData>(0.5)) {
        
        return result * clue::math::sqrt(TData{M_PI});
        }
        return result;
      }



    template <typename TData>
    ALPAKA_FN_ACC inline TData cyl_bessel_j_impl(TData nu, TData x) {
      if (x == static_cast<TData>(0)) {
        return (nu == static_cast<TData>(0)) ? static_cast<TData>(1) : static_cast<TData>(0);
      }

      const TData half_x = static_cast<TData>(0.5) * x;
      const TData x2 = x * x;
      const TData x4 = x2 * x2;
      const TData x6 = x4 * x2;
      const TData x8 = x4 * x4;

      const TData nu1 = nu + static_cast<TData>(1);
      const TData nu2 = nu + static_cast<TData>(2);
      const TData nu3 = nu + static_cast<TData>(3);
      const TData nu4 = nu + static_cast<TData>(4);

      const TData p_nu = static_cast<TData>(1)
                     - x4 / (static_cast<TData>(32) * nu1 * nu1 * nu2)
                     - x6 / (static_cast<TData>(96) * nu1 * nu1 * nu1 * nu2 * nu3)
                     + (nu - static_cast<TData>(5)) * x8 /
                           (static_cast<TData>(2048) * nu1 * nu1 * nu1 * nu1 * nu2 * nu3 * nu4);

      const TData gamma_nu1 = gamma(nu1);
      const TData decay = std::exp(-x2 / (static_cast<TData>(4) * nu1));

      return std::pow(half_x, nu) * p_nu / gamma_nu1 * decay;
    }

  }


template <typename TData>
ALPAKA_FN_ACC inline double cyl_bessel_j(TData n, TData x) {
    return detail::cyl_bessel_j_impl<double>(n, x);
}

}



 /* ALPAKA_FN_ACC inline double cyl_bessel_j(int n, double x) {
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
} */
  // namespace clue::math