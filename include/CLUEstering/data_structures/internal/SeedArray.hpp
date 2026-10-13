
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/data_structures/internal/DeviceVector.hpp"

namespace clue::inline CLUE_BACKEND::internal {

  template <clue::concepts::device TDev = clue::Device>
  using SeedArray = DeviceVector<TDev>;

  using SeedArrayView = DeviceVectorView;

}  // namespace clue::inline CLUE_BACKEND::internal
