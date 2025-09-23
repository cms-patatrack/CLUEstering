
#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/extrema/extrema.hpp"
#include <limits>
#include <span>

namespace clue::internal {

  template <clue::concepts::queue TQueue>
  inline auto make_associator(TQueue& queue, std::span<int32_t> associations, int32_t elements) {
    const auto bins = *clue::internal::algorithm::max_element(
                          associations.data(), associations.data() + associations.size()) +
                      1;
    clue::AssociationMap<decltype(alpaka::getDev(queue))> map(elements, bins, queue);
    map.template fill<clue::internal::Acc>(elements, associations, queue);
    return map;
  }

  inline auto make_associator(std::span<int32_t> associations, int32_t elements) {
    const auto bins = std::reduce(associations.data(),
                                  associations.data() + associations.size(),
                                  std::numeric_limits<int32_t>::lowest(),
                                  [](auto a, auto b) { return std::max(a, b); }) +
                      1;
    clue::host_associator map(elements, bins);
    map.fill(elements, associations);
    return map;
  }

}  // namespace clue::internal
