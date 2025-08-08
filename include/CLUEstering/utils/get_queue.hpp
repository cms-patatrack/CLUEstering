
#pragma once

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <std::integral TIdx>
  inline clue::Queue get_queue(TIdx device_id = TIdx{}) {
    auto device = alpaka::getDevByIdx(clue::Platform{}, device_id);
    return clue::Queue{device};
  }

  template <detail::concepts::device TDevice>
  inline clue::Queue get_queue(const TDevice& device) {
    return clue::Queue{device};
  }

}  // namespace clue
