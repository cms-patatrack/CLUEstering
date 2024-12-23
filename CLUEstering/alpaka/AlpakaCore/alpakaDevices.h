
#pragma once

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "alpakaConfig.h"
#include "getDeviceIndex.h"

namespace clue {

  // alpaka host device
  inline const alpaka_common::DevHost host =
      alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

  // alpaka accelerator devices
  template <typename TPlatform>
  inline std::vector<alpaka::Dev<TPlatform>> devices;

  template <typename TPlatform>
  std::vector<alpaka::Dev<TPlatform>> enumerate() {
    assert(getDeviceIndex(host) == 0u);

    using Device = alpaka::Dev<TPlatform>;
    using Platform = TPlatform;

    std::vector<Device> devices;
    uint32_t n = alpaka::getDevCount<Platform>();
    devices.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      devices.push_back(alpaka::getDevByIdx<Platform>(i));
      assert(getDeviceIndex(devices.back()) == static_cast<int>(i));
    }
    return devices;
  }

}  // namespace clue
