
#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/alpaka.hpp>

class FlatKernel {
private:
  float m_flat;

public:
  // Constructors
  FlatKernel() = delete;
  FlatKernel(float flat) : m_flat{flat} {}

  // Overload call operator
  template <typename TAcc>
  ALPAKA_FN_HOST_ACC float operator()(const TAcc&,
                                      float /*dist_ij*/,
                                      int point_id,
                                      int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_flat;
    }
  }
};

class GaussianKernel {
private:
  float m_gaus_avg;
  float m_gaus_std;
  float m_gaus_amplitude;

public:
  // Constructors
  GaussianKernel() = delete;
  GaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude)
      : m_gaus_avg{gaus_avg}, m_gaus_std{gaus_std}, m_gaus_amplitude{gaus_amplitude} {}

  // Overload call operator
  template <typename TAcc>
  ALPAKA_FN_HOST_ACC float operator()(const TAcc& acc,
                                      float dist_ij,
                                      int point_id,
                                      int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return (m_gaus_amplitude *
              alpaka::math::exp(acc,
                                -(dist_ij - m_gaus_avg) * (dist_ij - m_gaus_avg) /
                                    (2 * m_gaus_std * m_gaus_std)));
    }
  }
};

class ExponentialKernel {
private:
  float m_exp_avg;
  float m_exp_amplitude;

public:
  // Constructors
  ExponentialKernel() = delete;
  ExponentialKernel(float exp_avg, float exp_amplitude)
      : m_exp_avg{exp_avg}, m_exp_amplitude{exp_amplitude} {}

  // Overload call operator
  template <typename TAcc>
  ALPAKA_FN_HOST_ACC float operator()(const TAcc& acc,
                                      float dist_ij,
                                      int point_id,
                                      int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return (m_exp_amplitude * alpaka::math::exp(acc, -m_exp_avg * dist_ij));
    }
  }
};
