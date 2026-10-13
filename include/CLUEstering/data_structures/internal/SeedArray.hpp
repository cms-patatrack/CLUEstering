
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/data_structures/internal/DeviceVector.hpp"

CLUE_NAMESPACE_BEGIN
namespace internal {

  template <clue::concepts::device TDev = clue::Device>
  using SeedArray = DeviceVector<TDev>;

  using SeedArrayView = DeviceVectorView;

}  // namespace internal
CLUE_NAMESPACE_END
