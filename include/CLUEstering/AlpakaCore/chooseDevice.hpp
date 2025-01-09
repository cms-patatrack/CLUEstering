
#pragma once

#include "alpakaConfig.h"
#include "alpakaDevices.h"

namespace clue {

  template <typename TPlatform>
  alpaka::Dev<TPlatform> const& chooseDevice(edm::StreamID id) {
    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).

    // TODO: improve the "assignment" logic
    auto const& devices = clue::devices<TPlatform>;
    return devices[id % devices.size()];
  }

}  // namespace clue
