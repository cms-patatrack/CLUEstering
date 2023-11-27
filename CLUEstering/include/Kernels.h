/// \file Kernels.h
/// \brief Header file for the kernels used in the calculation of local density
///

#ifndef kernels_h
#define kernels_h

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using kernel_t = std::function<float(float, int, int)>;

/// @brief Class that represents the kernels used in the calculation of local density
/// @details The user can choose between a flat kernel, a gaussian kernel, an exponential
/// kernel or a custom kernel
/// @note The user can use a generic kernel by passing a function object
class kernel {
public:
  /// @brief Constructor
  kernel() = default;

  /// @brief Evaluate the kernel
  /// @param dist_ij distance between the points i and j
  /// @param point_id id of the point i
  /// @param j id of the point j
  /// @return the value of the kernel
  virtual float operator()(float dist_ij, int point_id, int j) const { return 0.f; };
};

/// @brief Class that represents the flat kernel
/// @details The flat kernel is defined as:
/// \f[
///  K_{ij} = \begin{cases}
///  1 & \text{if } i = j \\
///  \text{flat} & \text{if } i \neq j
///  \end{cases}
///  \f]
///  where \f$ \text{flat} \f$ is a user-defined parameter
///  @note The user can choose the value of \f$ \text{flat} \f$
///  @note The flat kernel is the default kernel
class flatKernel : public kernel {
private:
  float m_flat;

public:
  /// @brief Constructor
  /// @param flat value of the parameter \f$ \text{flat} \f$
  flatKernel(float flat) : m_flat{flat} {}

  /// @brief Evaluate the flat kernel
  /// @param dist_ij distance between the points i and j
  /// @param point_id id of the point i
  /// @param j id of the point j
  /// @return the value of the flat kernel
  float operator()(float dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_flat;
    }
  }
};

/// @brief Class that represents the gaussian kernel
/// @details The gaussian kernel is defined as:
/// \f[
/// K_{ij} = \begin{cases}
/// 1 & \text{if } i = j \\
/// \text{gaus_amplitude} \cdot \exp \left( - \frac{(dist_{ij} - \text{gaus_avg})^2}{2 \cdot \text{gaus_std}^2} \right) & \text{if } i \neq j
/// \end{cases}
/// \f]
/// where \f$ \text{gaus_avg} \f$, \f$ \text{gaus_std} \f$ and \f$ \text{gaus_amplitude} \f$ are user-defined parameters
/// @note The user can choose the values of \f$ \text{gaus_avg} \f$, \f$ \text{gaus_std} \f$ and \f$ \text{gaus_amplitude} \f$
class gaussianKernel : public kernel {
private:
  float m_gaus_avg;
  float m_gaus_std;
  float m_gaus_amplitude;

public:
  /// @brief Constructor
  /// @param gaus_avg value of the parameter \f$ \text{gaus_avg} \f$
  /// @param gaus_std value of the parameter \f$ \text{gaus_std} \f$
  /// @param gaus_amplitude value of the parameter \f$ \text{gaus_amplitude} \f$
  gaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude)
      : m_gaus_avg{gaus_avg}, m_gaus_std{gaus_std}, m_gaus_amplitude{gaus_amplitude} {}

  /// @brief Evaluate the gaussian kernel
  /// @param dist_ij distance between the points i and j
  /// @param point_id id of the point i
  /// @param j id of the point j
  /// @return the value of the gaussian kernel
  float operator()(float dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return static_cast<float>(
          m_gaus_amplitude * std::exp(-std::pow(dist_ij - m_gaus_avg, 2) / (2 * std::pow(m_gaus_std, 2))));
    }
  }
};

/// @brief Class that represents the exponential kernel
/// @details The exponential kernel is defined as:
/// \f[
/// K_{ij} = \begin{cases}
/// 1 & \text{if } i = j \\
/// \text{exp_amplitude} \cdot \exp \left( - \text{exp_avg} \cdot dist_{ij} \right) & \text{if } i \neq j
/// \end{cases}
/// \f]
/// where \f$ \text{exp_avg} \f$ and \f$ \text{exp_amplitude} \f$ are user-defined parameters
/// @note The user can choose the values of \f$ \text{exp_avg} \f$ and \f$ \text{exp_amplitude} \f$
class exponentialKernel : public kernel {
private:
  float m_exp_avg;
  float m_exp_amplitude;

public:
  /// @brief Constructor
  /// @param exp_avg value of the parameter \f$ \text{exp_avg} \f$
  /// @param exp_amplitude value of the parameter \f$ \text{exp_amplitude} \f$
  exponentialKernel(float exp_avg, float exp_amplitude)
      : m_exp_avg{exp_avg}, m_exp_amplitude{exp_amplitude} {}

  /// @brief Evaluate the exponential kernel
  /// @param dist_ij distance between the points i and j
  /// @param point_id id of the point i
  /// @param j id of the point j
  /// @return the value of the exponential kernel
  float operator()(float dist_ij, int point_id, int j) const override {
    if (point_id == j) {
      return 1.f;
    } else {
      return static_cast<float>(m_exp_amplitude * std::exp(-m_exp_avg * dist_ij));
    }
  }
};

/// @brief Class that represents a custom kernel
/// @details The user can pass a function object to the constructor
/// @note The function object must have the following signature:
/// \code{.cpp}
/// float function(float dist_ij, int point_id, int j);
/// \endcode
/// where \f$ dist_{ij} \f$ is the distance between the points i and j, \f$ point_{id} \f$ is the id of the point i and
/// \f$ j \f$ is the id of the point j
class customKernel : public kernel {
private:
  kernel_t m_function;

public:
  /// @brief Constructor
  /// @param function function object
  /// @note The function object must have the following signature:
  /// \code{.cpp}
  /// float function(float dist_ij, int point_id, int j);
  /// \endcode
  /// where \f$ dist_{ij} \f$ is the distance between the points i and j, \f$ point_{id} \f$ is the id of the point i
  /// and \f$ j \f$ is the id of the point j
  customKernel(kernel_t function) : m_function{std::move(function)} {}

  /// @brief Evaluate the custom kernel
  /// @param dist_ij distance between the points i and j
  /// @param point_id id of the point i
  /// @param j id of the point j
  /// @return the value of the custom kernel
  float operator()(float dist_ij, int point_id, int j) const override {
    return m_function(dist_ij, point_id, j);
  }
};

#endif
