
#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/reduce/reduce.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include <limits>
#include <span>

namespace clue::internal {

  template <clue::concepts::queue TQueue>
  inline auto make_associator(TQueue& queue,
                              std::span<const int32_t> associations,
                              int32_t elements) {
    const auto bins = clue::internal::algorithm::reduce(associations.begin(),
                                                        associations.end(),
                                                        std::numeric_limits<int32_t>::lowest(),
                                                        clue::nostd::maximum<int32_t>{}) +
                      1;
    clue::AssociationMap<decltype(alpaka::getDev(queue))> map(elements, bins, queue);
    map.template fill<clue::internal::Acc>(elements, associations, queue);
    alpaka::wait(queue);
    return map;
  }

  inline auto make_associator(std::span<const int32_t> associations, int32_t elements)
      -> AssociationMap<alpaka::DevCpu> {
    const auto bins = std::reduce(associations.begin(),
                                  associations.end(),
                                  std::numeric_limits<int32_t>::lowest(),
                                  clue::nostd::maximum<int32_t>{}) +

                      1;
    clue::host_associator map(elements, bins);
    map.fill(associations);
    return map;
  }

}  // namespace clue::internal
