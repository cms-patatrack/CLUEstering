
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <alpaka/alpaka.hpp>

namespace clue {

  inline FlatKernel::FlatKernel(float flat) : m_flat{flat} {}

  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC float FlatKernel::operator()(const TAcc&,
                                                         float /*dist_ij*/,
                                                         int point_id,
                                                         int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_flat;
    }
  }

  inline GaussianKernel::GaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude)
      : m_gaus_avg{gaus_avg}, m_gaus_std{gaus_std}, m_gaus_amplitude{gaus_amplitude} {}

  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC float GaussianKernel::operator()(const TAcc& acc,
                                                             float dist_ij,
                                                             int point_id,
                                                             int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_gaus_amplitude *
             clue::internal::math::exp(-(dist_ij - m_gaus_avg) * (dist_ij - m_gaus_avg) /
                                       (2 * m_gaus_std * m_gaus_std));
    }
  }

  inline ExponentialKernel::ExponentialKernel(float exp_avg, float exp_amplitude)
      : m_exp_avg{exp_avg}, m_exp_amplitude{exp_amplitude} {}

  template <typename TAcc>
  inline ALPAKA_FN_HOST_ACC float ExponentialKernel::operator()(const TAcc& acc,
                                                                float dist_ij,
                                                                int point_id,
                                                                int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return (m_exp_amplitude * clue::internal::math::exp(-m_exp_avg * dist_ij));
    }
  }

}  // namespace clue
