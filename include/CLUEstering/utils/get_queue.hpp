/// @file get_queue.hpp
/// @brief Provides functions to get an alpaka queue from a device index or a device object
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

namespace clue {

  /// @brief Get an alpaka queue created from a device correspoding to a given index
  ///
  /// @tparam TIdx The type of the device index, must be an integral type
  /// @param device_id The index of the device
  /// @return An alpaka queue created from the device corresponding to the given index
  template <std::integral TIdx>
  inline clue::Queue get_queue(TIdx device_id = TIdx{}) {
    auto device = alpaka::getDevByIdx(clue::Platform{}, device_id);
    return clue::Queue{device};
  }

  /// @brief Get an alpaka queue created from a given device
  ///
  /// @tparam TDevice The type of the device, must satisfy the `concepts::device` concept
  /// @param device The device to create the queue from
  /// @return An alpaka queue created from the given device
  template <concepts::device TDevice>
  inline clue::Queue get_queue(const TDevice& device) {
    return clue::Queue{device};
  }

}  // namespace clue
