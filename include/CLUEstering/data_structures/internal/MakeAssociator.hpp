
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/reduce/reduce.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include <limits>
#include <span>
#include <numeric>

CLUE_NAMESPACE_BEGIN
namespace internal {

  template <typename T>
  struct non_negative {
    ALPAKA_FN_HOST_ACC constexpr auto operator()(T value) const { return value >= 0; }
  };

  template <clue::concepts::queue TQueue, clue::concepts::allocator TAllocator>
  inline auto make_associator(TQueue& queue,
                              std::span<const int32_t> associations,
                              const TAllocator& allocator) {
	auto nelements = std::ranges::count_if(queue,
										   associations.begin(),
										   associations.end(),
										   non_negative<int32_t>{});
    const auto nbins = clue::internal::algorithm::reduce(queue,
														 associations.begin(),
														 associations.end(),
														 std::numeric_limits<int32_t>::lowest(),
														 clue::nostd::maximum<int32_t>{}) +
	  1;

    clue::AssociationMap<decltype(alpaka::getDev(queue))> map(nelements, nbins, queue, allocator);
    map.template fill<clue::internal::Acc>(associations, queue, allocator);
    alpaka::wait(queue);
    return map;
  }

  inline auto make_associator(std::span<const int32_t> associations)
      -> AssociationMap<alpaka::DevCpu> {
	const auto nelements = std::ranges::count_if(associations, non_negative<int32_t>{});
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
