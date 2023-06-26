#ifndef convolutional_kernels_h
#define convolutional_kernels_h

#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

using kernel_t = std::function<float(float, int, int)>;

struct ConvolutionalKernel {
  ConvolutionalKernel() = default;

  template <typename TAcc>
  virtual ALPAKA_FN_ACC auto operator()(const TAcc &acc, double dist_ij, int point_id, int j) const {
    return 0.f;
  }
};

struct FlatKernel : ConvolutionalKernel {
  float m_flat;
  FlatKernel() = delete;
  FlatKernel(float flat) : m_flat{flat} {}

  template <typename TAcc>
  ALPAKA_FN_ACC auto operator()(const TAcc &acc, double dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_flat;
    }
  }
};

struct GaussianKernel : ConvolutionalKernel {
  float m_gaus_amplitude;
  float m_gaus_avg;
  float m_gaus_std;
  GaussianKernel() = delete;
  GaussianKernel(float gaus_amplitude, float gaus_avg, float gaus_std)
      : m_gaus_amplitude{gaus_amplitude}, m_gaus_avg{gaus_avg}, m_gaus_std{gaus_std} {}

  template <typename TAcc>
  ALPAKA_FN_ACC auto operator()(const TAcc &acc, float dist_ij, int point_id, int j) const override {
    return (m_gaus_amplitude * alpaka::math::exp(-alpaka::math::pow(dist_ij - m_gaus_avg, 2) / (2 * alpaka::math::pow(m_gaus_std, 2))));
  }
};

struct ExponentialKernel : ConvolutionalKernel {
  float m_exp_amplitude;
  float m_exp_avg;
  ExponentialKernel() = delete;
  ExponentialKernel(float exp_amplitude, float exp_avg) : m_exp_amplitude{exp_amplitude}, m_exp_avg{exp_avg} {}

  template <typename TAcc>
  ALPAKA_FN_ACC auto operator()(const TAcc &acc, float dist_ij, int point_id, int j) const override {
    return (m_exp_amplitude * alpaka::math::exp(-m_exp_avg * dist_ij))
  }
};

struct CustomKernel : ConvolutionalKernel {
  kernel_t m_function;
  customKernel() = delete;
  customKernel(kernel_t function) : m_function{std::move(function)} {}

  template <typename TAcc>
  ALPAKA_FN_ACC auto operator()(const TAcc &acc, float dist_ij, int point_id, int j) const override {
    return m_function(dist_ij, point_id, j);
  }
};

#endif