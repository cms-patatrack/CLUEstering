
#pragma once

#include "CLUEstering/core/defines.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/extrema/extrema.hpp"
#include <span>

namespace clue {
  namespace test {

    template <clue::detail::concepts::queue TQueue>
    inline auto build_map(TQueue& queue, std::span<int32_t> associations, int32_t elements) {
      const auto bins = *clue::internal::algorithm::max_element(
                            associations.data(), associations.data() + associations.size()) +
                        1;
      clue::AssociationMap<decltype(alpaka::getDev(queue))> map(elements, bins, queue);
      map.template fill<clue::internal::Acc>(elements, associations, queue);
      return map;
    }

  }  // namespace test
}  // namespace clue
