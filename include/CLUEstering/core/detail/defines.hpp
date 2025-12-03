
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"

namespace clue {

  using Platform = ALPAKA_BACKEND::Platform;
  using Device = ALPAKA_BACKEND::Device;
  using Queue = ALPAKA_BACKEND::Queue;
  using Event = ALPAKA_BACKEND::Event;

  namespace internal {

    using namespace alpaka_common;
    using Acc = ALPAKA_BACKEND::Acc1D;
    using Acc2D = ALPAKA_BACKEND::Acc2D;

  }  // namespace internal

}  // namespace clue
