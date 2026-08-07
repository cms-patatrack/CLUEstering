
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <concepts>
#include <stdexcept>
#include <alpaka/alpaka.hpp>

namespace clue {


  

  template <std::floating_point TData, std::size_t Ndim>
inline FlatKernel<TData, Ndim>::FlatKernel(value_type flat) 
    : m_flat{flat}, density_radius{value_type{4}} {
  if (flat <= value_type{0}) {
    throw std::invalid_argument("Flat kernel parameter must be positive.");
  }
}

  template <std::floating_point TData, std::size_t Ndim>
  inline FlatKernel<TData, Ndim>::FlatKernel(value_type flat, value_type radius) 
      : m_flat{flat}, density_radius{radius} {
    if (flat <= value_type{0}) {
      throw std::invalid_argument("Flat kernel parameter must be positive.");
    }
  }

  template <std::floating_point TData, std::size_t Ndim>
  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC auto FlatKernel<TData, Ndim>::operator()(const TAcc&,
                                                               value_type /*dist_ij*/,
                                                               int point_id,
                                                               int j) const {
    if (point_id == j) {
      return value_type{1};
    } else {
      return m_flat;
    }
  }

  template <std::floating_point TData, std::size_t Ndim>
  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC auto FlatKernel<TData, Ndim>::freq_eval(const TAcc&,
                                                              const value_type* k) const {
    const auto prefactor = math::pow(value_type{2.0} * value_type{M_PI}, value_type(Ndim) / value_type{2}) * math::pow(density_radius, value_type(Ndim));
  
    auto k_norm = value_type{0};
    for (int i = 0; i < Ndim; ++i) {
      k_norm += k[i] * k[i];
    }
    /// 
    k_norm = math::sqrt(k_norm);
    return m_flat * prefactor * math::pow(density_radius * k_norm, - value_type(Ndim) / value_type{2}) * math::cyl_bessel_j(value_type(Ndim) / value_type{2}, density_radius * k_norm);
    
  }

  template <std::floating_point TData, std::size_t Ndim>
  inline GaussianKernel<TData, Ndim>::GaussianKernel(value_type gaus_avg,
                                               value_type gaus_std,
                                               value_type gaus_amplitude)
      : m_gaus_avg{gaus_avg}, m_gaus_std{gaus_std}, m_gaus_amplitude{gaus_amplitude} {
    if (gaus_std <= value_type{0} || gaus_amplitude <= value_type{0}) {
      throw std::invalid_argument("Gaussian kernel parameters must be positive.");
    }
  }

  template <std::floating_point TData, std::size_t Ndim>
  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC auto GaussianKernel<TData, Ndim>::operator()(const TAcc&,
                                                                   value_type dist_ij,
                                                                   int point_id,
                                                                   int j) const {
    if (point_id == j) {
      return value_type{1};
    } else {
      return m_gaus_amplitude * math::exp(-(dist_ij - m_gaus_avg) * (dist_ij - m_gaus_avg) /
                                          (2 * m_gaus_std * m_gaus_std));
    }
  }

  template <std::floating_point TData, std::size_t Ndim>
  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC auto GaussianKernel<TData, Ndim>::freq_eval(const TAcc&,
                                                                  const value_type* k) const {

   auto k_norm = value_type{0};
    for (int i = 0; i < Ndim; ++i) {
      k_norm += k[i] * k[i];
    }
    k_norm = math::sqrt(k_norm);                                                                  
  const auto prefactor = math::pow(value_type{2.0} * value_type{M_PI}, value_type(Ndim) / value_type{2});
  const auto sigma_n = math::pow(m_gaus_std, value_type(Ndim));  
  return m_gaus_amplitude * prefactor * sigma_n *
         math::exp(value_type{-0.5} * m_gaus_std * m_gaus_std * k_norm * k_norm);
}



  template <std::floating_point TData, std::size_t Ndim>
  inline ExponentialKernel<TData, Ndim>::ExponentialKernel(value_type exp_avg, value_type exp_amplitude)
      : m_exp_avg{exp_avg}, m_exp_amplitude{exp_amplitude} {
    if (exp_avg <= value_type{0} || exp_amplitude <= value_type{0}) {
      throw std::invalid_argument("Exponential kernel parameters must be positive.");
    }
  }

  template <std::floating_point TData, std::size_t Ndim>
  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC auto ExponentialKernel<TData, Ndim>::operator()(const TAcc&,
                                                                      value_type dist_ij,
                                                                      int point_id,
                                                                      int j) const {
    if (point_id == j) {
      return value_type{1};
    } else {
      return (m_exp_amplitude * math::exp(-m_exp_avg * dist_ij));
    }
  }

  template <std::floating_point TData, std::size_t Ndim>
  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC auto ExponentialKernel<TData, Ndim>::freq_eval(const TAcc&,
                                                                     const value_type* k) const {
    const auto prefactor = math::pow(value_type{2},Ndim) * math::pow(value_type{M_PI}, value_type(Ndim) / value_type{2});
    return prefactor * (1/m_exp_avg)* math::tgamma((value_type(Ndim)+value_type{2})/value_type{2});
  }

}  // namespace clue
