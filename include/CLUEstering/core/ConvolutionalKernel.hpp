
#pragma once

#include <alpaka/alpaka.hpp>

namespace clue {

  class FlatKernel {
  private:
    float m_flat;

  public:
    FlatKernel(float flat);

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC float operator()(const TAcc&, float /*dist_ij*/, int point_id, int j) const;
  };

  class GaussianKernel {
  private:
    float m_gaus_avg;
    float m_gaus_std;
    float m_gaus_amplitude;

  public:
    GaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude);

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC float operator()(const TAcc& acc, float dist_ij, int point_id, int j) const;
  };

  class ExponentialKernel {
  private:
    float m_exp_avg;
    float m_exp_amplitude;

  public:
    ExponentialKernel(float exp_avg, float exp_amplitude);

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC float operator()(const TAcc& acc, float dist_ij, int point_id, int j) const;
  };

}  // namespace clue

#include "CLUEstering/core/detail/ConvolutionalKernel.hpp"
