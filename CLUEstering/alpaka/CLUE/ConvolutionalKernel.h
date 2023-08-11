#ifndef convolutional_kernels_h
#define convolutional_kernels_h

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Debug.hpp>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

using kernel_t = std::function<float(float, int, int)>;

class ConvolutionalKernel {
public:
  // Constructor
  ConvolutionalKernel() = default;

  // Overload call operator
  virtual ALPAKA_FN_HOST_ACC float operator()(float dist_ij, int point_id, int j) const { return 0.f; }

  kernel_t extract() const {
	kernel_t l = [this](float dist, int i, int j) {
	  return (*this)(dist, i, j);
	};

	return l;
  }
};

class FlatKernel : public ConvolutionalKernel {
private:
  float m_flat;

public:
  // Constructors
  FlatKernel() = delete;
  FlatKernel(float flat) : m_flat{flat} {}

  // Overload call operator
  ALPAKA_FN_HOST_ACC float operator()(float dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_flat;
    }
  }
};

struct GaussianKernel : public ConvolutionalKernel {
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
  ALPAKA_FN_HOST_ACC float operator()(float dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return (m_gaus_amplitude *
              std::exp(-std::pow(dist_ij - m_gaus_avg, 2) / (2 * m_gaus_std * m_gaus_std)));
    }
  }
};

struct ExponentialKernel : public ConvolutionalKernel {
private:
  float m_exp_avg;
  float m_exp_amplitude;

public:
  // Constructors
  ExponentialKernel() = delete;
  ExponentialKernel(float exp_avg, float exp_amplitude)
      : m_exp_avg{exp_avg}, m_exp_amplitude{exp_amplitude} {}

  // Overload call operator
  ALPAKA_FN_HOST_ACC float operator()(float dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return (m_exp_amplitude * std::exp(-m_exp_avg * dist_ij));
    }
  }
};

struct CustomKernel : public ConvolutionalKernel {
private:
  kernel_t m_function;

public:
  // Constructors
  CustomKernel() = delete;
  CustomKernel(kernel_t function) : m_function{std::move(function)} {}

  // Overload call operator
  ALPAKA_FN_HOST_ACC float operator()(float dist_ij, int point_id, int j) const override {
    return m_function(dist_ij, point_id, j);
  }
};

#endif
