
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <concepts>
#include <stdexcept>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <std::floating_point TData>
  inline FlatKernel<TData>::FlatKernel(value_type flat) : m_flat{flat} {
    if (flat <= value_type{0}) {
      throw std::invalid_argument("Flat kernel parameter must be positive.");
    }
  }

  template <std::floating_point TData>
  inline ALPAKA_FN_HOST_ACC auto FlatKernel<TData>::operator()(value_type /*dist_ij*/,
                                                               int point_id,
                                                               int j) const {
    return (point_id == j) ? value_type{1} : m_flat;
  }

  template <std::floating_point TData>
  inline GaussianKernel<TData>::GaussianKernel(value_type gaus_std, value_type gaus_amplitude)
      : m_gaus_std{gaus_std}, m_gaus_amplitude{gaus_amplitude} {
    if (gaus_std <= value_type{0} || gaus_amplitude <= value_type{0}) {
      throw std::invalid_argument("Gaussian kernel parameters must be positive.");
    }
  }

  template <std::floating_point TData>
  inline ALPAKA_FN_HOST_ACC auto GaussianKernel<TData>::operator()(value_type dist_ij,
                                                                   int point_id,
                                                                   int j) const {
    return (point_id == j)
               ? value_type{1}
               : m_gaus_amplitude * math::exp(-dist_ij * dist_ij / (2 * m_gaus_std * m_gaus_std));
  }

  template <std::floating_point TData>
  inline ExponentialKernel<TData>::ExponentialKernel(value_type exp_avg, value_type exp_amplitude)
      : m_exp_avg{exp_avg}, m_exp_amplitude{exp_amplitude} {
    if (exp_avg <= value_type{0} || exp_amplitude <= value_type{0}) {
      throw std::invalid_argument("Exponential kernel parameters must be positive.");
    }
  }

  template <std::floating_point TData>
  inline ALPAKA_FN_HOST_ACC auto ExponentialKernel<TData>::operator()(value_type dist_ij,
                                                                      int point_id,
                                                                      int j) const {
    return (point_id == j) ? value_type{1} : (m_exp_amplitude * math::exp(-m_exp_avg * dist_ij));
  }

}  // namespace clue
