/// @file ConvolutionalKernel.hpp
/// @brief Provides the kernel classes for the convolution done when computing the weighted density of the points
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include <alpaka/alpaka.hpp>
#include <concepts>

namespace clue {

  /// @brief The FlatKernel class implements a flat kernel for convolution.
  /// It returns a constant value for the kernel, regardless of the distance between points.
  ///
  /// @tparam TData The data type for the kernel values
  template <std::floating_point TData = float>
  class FlatKernel {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

  private:
    value_type m_flat;

  public:
    /// @brief Construct a FlatKernel object
    ///
    /// @param flat The flat value for the kernel
    FlatKernel(value_type flat);

    /// @brief Computes the kernel value between two points
    ///
    /// @param acc The accelerator to use for the computation
    /// @param dist_ij The distance between the two points
    /// @param point_id The index of the first point
    /// @param j The index of the second point
    /// @return The computed kernel value
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(const TAcc& acc, value_type dist_ij, int point_id, int j) const;
  };

  /// @brief The GaussianKernel class implements a Gaussian kernel for convolution.
  /// It computes the kernel value based on the Gaussian function, which is defined by its average, standard deviation, and amplitude.
  ///
  /// @tparam TData The data type for the kernel values
  template <std::floating_point TData = float>
  class GaussianKernel {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

  private:
    value_type m_gaus_avg;
    value_type m_gaus_std;
    value_type m_gaus_amplitude;

  public:
    /// @brief Construct a GaussianKernel object
    ///
    /// @param gaus_avg The average value for the Gaussian kernel
    /// @param gaus_std The standard deviation for the Gaussian kernel
    /// @param gaus_amplitude The amplitude for the Gaussian kernel
    GaussianKernel(value_type gaus_avg, value_type gaus_std, value_type gaus_amplitude);

    /// @brief Computes the kernel value between two points
    ///
    /// @param acc The accelerator to use for the computation
    /// @param dist_ij The distance between the two points
    /// @param point_id The index of the first point
    /// @param j The index of the second point
    /// @return The computed kernel value
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(const TAcc& acc, value_type dist_ij, int point_id, int j) const;
  };

  /// @brief The ExponentialKernel class implements an exponential kernel for convolution.
  /// It computes the kernel value based on the exponential function, which is defined by its average and amplitude.
  ///
  /// @tparam TData The data type for the kernel values
  template <std::floating_point TData = float>
  class ExponentialKernel {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

  private:
    value_type m_exp_avg;
    value_type m_exp_amplitude;

  public:
    /// @brief Construct an ExponentialKernel object
    ///
    /// @param exp_avg The average value for the exponential kernel
    /// @param exp_amplitude The amplitude for the exponential kernel
    ExponentialKernel(value_type exp_avg, value_type exp_amplitude);

    /// @brief Computes the kernel value between two points
    ///
    /// @param acc The accelerator to use for the computation
    /// @param dist_ij The distance between the two points
    /// @param point_id The index of the first point
    /// @param j The index of the second point
    /// @return The computed kernel value
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(const TAcc& acc, value_type dist_ij, int point_id, int j) const;
  };

  namespace concepts {

    /// @brief Concept describing a convolutional kernel
    template <typename TKernel>
    concept convolutional_kernel =
        requires(TKernel&& kernel,
                 const internal::Acc& acc,
                 typename std::remove_cvref_t<TKernel>::value_type distance,
                 int point_i,
                 int point_j) {
          {
            kernel(acc, distance, point_i, point_j)
          } -> std::same_as<typename std::remove_cvref_t<TKernel>::value_type>;
        };

  }  // namespace concepts

}  // namespace clue

#include "CLUEstering/core/detail/ConvolutionalKernel.hpp"
