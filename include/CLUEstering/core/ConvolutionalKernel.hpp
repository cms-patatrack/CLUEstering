/// @file ConvolutionalKernel.hpp
/// @brief Provides the kernel classes for the convolution done when computing the weighted density of the points
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include <alpaka/alpaka.hpp>

namespace clue {

  /// @brief The FlatKernel class implements a flat kernel for convolution.
  /// It returns a constant value for the kernel, regardless of the distance between points.
  class FlatKernel {
  private:
    float m_flat;

  public:
    /// @brief Construct a FlatKernel object
    ///
    /// @param flat The flat value for the kernel
    FlatKernel(float flat);

    /// @brief Computes the kernel value between two points
    ///
    /// @param acc The accelerator to use for the computation
    /// @param dist_ij The distance between the two points
    /// @param point_id The index of the first point
    /// @param j The index of the second point
    /// @return The computed kernel value
    template <typename TAcc>
    ALPAKA_FN_HOST_ACC float operator()(const TAcc&, float /*dist_ij*/, int point_id, int j) const;
  };

  /// @brief The GaussianKernel class implements a Gaussian kernel for convolution.
  /// It computes the kernel value based on the Gaussian function, which is defined by its average, standard deviation, and amplitude.
  class GaussianKernel {
  private:
    float m_gaus_avg;
    float m_gaus_std;
    float m_gaus_amplitude;

  public:
    /// @brief Construct a GaussianKernel object
    ///
    /// @param gaus_avg The average value for the Gaussian kernel
    /// @param gaus_std The standard deviation for the Gaussian kernel
    /// @param gaus_amplitude The amplitude for the Gaussian kernel
    GaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude);

    /// @brief Computes the kernel value between two points
    ///
    /// @param acc The accelerator to use for the computation
    /// @param dist_ij The distance between the two points
    /// @param point_id The index of the first point
    /// @param j The index of the second point
    /// @return The computed kernel value
    template <typename TAcc>
    ALPAKA_FN_HOST_ACC float operator()(const TAcc& acc, float dist_ij, int point_id, int j) const;
  };

  /// @brief The ExponentialKernel class implements an exponential kernel for convolution.
  /// It computes the kernel value based on the exponential function, which is defined by its average and amplitude.
  class ExponentialKernel {
  private:
    float m_exp_avg;
    float m_exp_amplitude;

  public:
    /// @brief Construct an ExponentialKernel object
    ///
    /// @param exp_avg The average value for the exponential kernel
    /// @param exp_amplitude The amplitude for the exponential kernel
    ExponentialKernel(float exp_avg, float exp_amplitude);

    /// @brief Computes the kernel value between two points
    ///
    /// @param acc The accelerator to use for the computation
    /// @param dist_ij The distance between the two points
    /// @param point_id The index of the first point
    /// @param j The index of the second point
    /// @return The computed kernel value
    template <typename TAcc>
    ALPAKA_FN_HOST_ACC float operator()(const TAcc& acc, float dist_ij, int point_id, int j) const;
  };

}  // namespace clue

#include "CLUEstering/core/detail/ConvolutionalKernel.hpp"
