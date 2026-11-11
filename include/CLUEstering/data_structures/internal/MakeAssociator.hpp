
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/reduce/reduce.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include <limits>
#include <span>

CLUE_NAMESPACE_BEGIN
namespace internal {

  template <clue::concepts::queue TQueue, clue::concepts::allocator TAllocator>
  inline auto make_associator(TQueue& queue,
                              std::span<const int32_t> associations,
                              int32_t nelements,
                              const TAllocator& allocator) {
    const auto nbins = clue::internal::algorithm::reduce(queue,
														 associations.begin(),
														 associations.end(),
														 std::numeric_limits<int32_t>::lowest(),
														 clue::nostd::maximum<int32_t>{}) +
                      1;
    alpaka::wait(queue);

    clue::AssociationMap<decltype(alpaka::getDev(queue))> map(nelements, nbins, queue, allocator);
    map.template fill<clue::internal::Acc>(associations, queue, allocator);
    alpaka::wait(queue);
    return map;
  }

  inline auto make_associator(std::span<const int32_t> associations, int32_t nelements)
      -> AssociationMap<alpaka::DevCpu> {
    const auto nbins = std::reduce(associations.begin(),
								   associations.end(),
								   std::numeric_limits<int32_t>::lowest(),
								   clue::nostd::maximum<int32_t>{}) +

                      1;
    clue::AssociationMapHost map(nelements, nbins);
    map.fill(associations);
    return map;
  }

}  // namespace internal
CLUE_NAMESPACE_END
