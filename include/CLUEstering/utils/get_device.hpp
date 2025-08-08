
#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include <alpaka/alpaka.hpp>

namespace clue {

  inline clue::Device get_device(uint32_t device_id) {
    return alpaka::getDevByIdx(clue::Platform{}, device_id);
  }

}  // namespace clue
