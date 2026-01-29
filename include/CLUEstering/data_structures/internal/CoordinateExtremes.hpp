
#pragma once

#include <array>
#include <alpaka/alpaka.hpp>

namespace clue::internal {

  template <std::size_t Ndim, typename TData = float>
  class CoordinateExtremes {
  private:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

    std::array<value_type, 2 * Ndim> m_data;

  public:
    CoordinateExtremes() = default;

    ALPAKA_FN_HOST_ACC const auto* data() const { return m_data; }
    ALPAKA_FN_HOST_ACC auto* data() { return m_data; }

    ALPAKA_FN_HOST_ACC auto min(int i) const { return m_data[2 * i]; }
    ALPAKA_FN_HOST_ACC auto& min(int i) { return m_data[2 * i]; }
    ALPAKA_FN_HOST_ACC auto max(int i) const { return m_data[2 * i + 1]; }
    ALPAKA_FN_HOST_ACC auto& max(int i) { return m_data[2 * i + 1]; }
    ALPAKA_FN_HOST_ACC auto range(int i) const { return max(i) - min(i); }
  };

}  // namespace clue::internal
