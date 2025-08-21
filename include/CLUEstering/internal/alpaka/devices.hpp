
#pragma once

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/get_device_index.hpp"
#include "CLUEstering/detail/concepts.hpp"

namespace clue {

  // returns the alpaka accelerator platform
  template <concepts::platform TPlatform>
  inline TPlatform const& platform() {
    // initialise the platform the first time that this function is called
    static const auto platform = TPlatform{};
    return platform;
  }

  // return the alpaka accelerator devices for the given platform
  template <concepts::platform TPlatform>
  inline std::vector<alpaka::Dev<TPlatform>> const& devices() {
    // enumerate all devices the first time that this function is called
    static const auto devices = alpaka::getDevs(platform<TPlatform>());
    return devices;
  }

  // alpaka host device
  inline const alpaka_common::DevHost host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

  // alpaka accelerator devices
  /* template <typename TPlatform> */
  /* inline std::vector<alpaka::Dev<TPlatform>> devices; */

  template <concepts::platform TPlatform>
  std::vector<alpaka::Dev<TPlatform>> enumerate() {
    assert(getDeviceIndex(host) == 0u);

    using Device = alpaka::Dev<TPlatform>;
    using Platform = TPlatform;

    std::vector<Device> devices;
    uint32_t n = alpaka::getDevCount(Platform{});
    devices.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      devices.push_back(alpaka::getDevByIdx(Platform{}, i));
      assert(getDeviceIndex(devices.back()) == static_cast<int>(i));
    }
    return devices;
  }

}  // namespace clue
