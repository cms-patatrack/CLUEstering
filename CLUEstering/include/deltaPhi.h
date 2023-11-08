/// \file deltaPhi.h
/// \brief Header file for the deltaPhi function
///

#include <algorithm>

/// @brief Calculate the difference in phi between two points
/// @tparam T type of the input parameters
/// @param phi1 phi of the first point
/// @param phi2 phi of the second point
/// @param x_min minimum value of phi
/// @param x_max maximum value of phi
/// @return the difference in phi between the two points
template <typename T>
T deltaPhi(T phi1, T phi2, T x_min, T x_max) {
  T delta_phi{std::abs(phi2 - phi1)};
  return std::min(delta_phi, x_max - x_min - delta_phi);
}
