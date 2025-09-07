
#pragma once

#include <execution>

namespace clue {
  namespace internal {

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    inline constexpr auto default_policy = std::execution::par_unseq;
#else
    inline constexpr auto default_policy = std::execution::unseq;
#endif

  }  // namespace internal
}  // namespace clue
