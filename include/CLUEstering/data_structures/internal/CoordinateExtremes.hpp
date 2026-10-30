
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <alpaka/alpaka.hpp>
#include <type_traits>

CLUE_NAMESPACE_BEGIN
namespace internal {

  template <std::size_t Ndim, typename TData = float>
  class CoordinateExtremes {
  private:
    using value_type = std::remove_cvref_t<TData>;

    std::array<value_type, 2 * Ndim> m_data;

  public:
    CoordinateExtremes() = default;

    ALPAKA_FN_HOST_ACC const auto* data() const { return m_data; }
    ALPAKA_FN_HOST_ACC auto* data() { return m_data; }

    ALPAKA_FN_HOST_ACC auto min(int dim) const {
      assert(dim >= 0 && static_cast<std::size_t>(dim) < Ndim);
      return m_data[2 * dim];
    }
    ALPAKA_FN_HOST_ACC auto& min(int dim) {
      assert(dim >= 0 && static_cast<std::size_t>(dim) < Ndim);
      return m_data[2 * dim];
    }
    ALPAKA_FN_HOST_ACC auto max(int dim) const {
      assert(dim >= 0 && static_cast<std::size_t>(dim) < Ndim);
      return m_data[2 * dim + 1];
    }
    ALPAKA_FN_HOST_ACC auto& max(int dim) {
      assert(dim >= 0 && static_cast<std::size_t>(dim) < Ndim);
      return m_data[2 * dim + 1];
    }
    ALPAKA_FN_HOST_ACC auto range(int dim) const { return max(dim) - min(dim); }
  };

}  // namespace internal
CLUE_NAMESPACE_END
