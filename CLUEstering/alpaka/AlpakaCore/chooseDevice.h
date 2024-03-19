#ifndef HeterogeneousCore_AlpakaCore_chooseDevice_h
#define HeterogeneousCore_AlpakaCore_chooseDevice_h

#include "alpakaConfig.h"
#include "alpakaDevices.h"

namespace cms::alpakatools {

  template <typename TPlatform>
  alpaka::Dev<TPlatform> const& chooseDevice(edm::StreamID id) {
    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).

    // TODO: improve the "assignment" logic
    auto const& devices = cms::alpakatools::devices<TPlatform>;
    return devices[id % devices.size()];
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_chooseDevice_h
