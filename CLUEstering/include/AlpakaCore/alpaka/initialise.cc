#include <iostream>

#include <alpaka/alpaka.hpp>

#include "../alpakaConfig.h"
#include "../alpakaDevices.h"
#include "../initialise.h"
#include "../demangle.h"

namespace clue {

  template <typename TPlatform>
  void initialise() {
    constexpr const char* suffix[] = {"devices.", "device:", "devices:"};

    if (devices<TPlatform>.empty()) {
      devices<TPlatform> = enumerate<TPlatform>();
      auto size = devices<TPlatform>.size();
      //std::cout << edm::demangle<TPlatform> << " platform succesfully initialised." << std::endl;
      std::cout << "Found " << size << " " << suffix[size < 2 ? size : 2] << std::endl;
      for (auto const& device : devices<TPlatform>) {
        std::cout << "  - " << alpaka::getName(device) << std::endl;
      }
    } else {
      //std::cout << edm::demangle<TPlatform> << " platform already initialised." << std::endl;
    }
  }

  // explicit template instantiation definition
  template void initialise<ALPAKA_ACCELERATOR_NAMESPACE::Platform>();

}  // namespace clue
