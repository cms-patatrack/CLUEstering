/// @file get_device.hpp
/// @brief Provides a function to get an alpaka device by its index
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include <alpaka/alpaka.hpp>

namespace clue {

  /// @brief Get the alpaka device corresponding to a given index
  ///
  /// @param device_id The index of the device
  /// @return The alpaka device corresponding to the given index
  inline clue::Device get_device(uint32_t device_id) {
    return alpaka::getDevByIdx(clue::Platform{}, device_id);
  }

}  // namespace clue
