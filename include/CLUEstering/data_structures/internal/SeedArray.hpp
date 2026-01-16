
#pragma once

#include "CLUEstering/data_structures/internal/DeviceVector.hpp"

namespace clue::internal {

  template <clue::concepts::device TDev = clue::Device>
  using SeedArray = DeviceVector<TDev>;

  using SeedArrayView = DeviceVectorView;

}  // namespace clue::internal
