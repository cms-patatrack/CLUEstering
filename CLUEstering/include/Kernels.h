#ifndef kernels_h
#define kernels_h

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using kernel_t = std::function<float(float, int, int)>;

// Define the kernels used for the calculation of local density
class kernel {
public:
  kernel() = default;

  virtual float operator()(float dist_ij, int point_id, int j) const { return 0.f; };
};

class flatKernel : public kernel {
private:
  float m_flat;

public:
  flatKernel(float flat) : m_flat(flat) {}

  float operator()(float dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_flat;
    }
  }
};

class gaussianKernel : public kernel {
private:
  float m_gaus_avg;
  float m_gaus_std;
  float m_gaus_amplitude;

public:
  gaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude)
      : m_gaus_avg(gaus_avg), m_gaus_std(gaus_std), m_gaus_amplitude(gaus_amplitude) {}

  float operator()(float dist_ij, int point_id, int j) const override {
    return static_cast<float>(m_gaus_amplitude *
                              std::exp(-std::pow(dist_ij - m_gaus_avg, 2) / (2 * std::pow(m_gaus_std, 2))));
  }
};

class exponentialKernel : public kernel {
private:
  float m_exp_avg;
  float m_exp_amplitude;

public:
  exponentialKernel(float exp_avg, float exp_amplitude) : m_exp_avg(exp_avg), m_exp_amplitude(exp_amplitude) {}

  float operator()(float dist_ij, int point_id, int j) const override {
    return static_cast<float>(m_exp_amplitude * exp(-m_exp_avg * dist_ij));
  }
};

class customKernel : public kernel {
private:
  kernel_t m_function;

public:
  customKernel(kernel_t function) : m_function(std::move(function)) {}

  float operator()(float dist_ij, int point_id, int j) const override { return m_function(dist_ij, point_id, j); }
};

#endif
