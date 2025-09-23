
#pragma once

#include <array>
#include <alpaka/alpaka.hpp>

namespace clue::internal {

  struct Extreme {
    float min;
    float max;
  };

  template <uint8_t Ndim>
  class CoordinateExtremes {
  private:
    std::array<Extreme, Ndim> m_data;

  public:
    ALPAKA_FN_HOST_ACC const float* data() const { return m_data; }
    ALPAKA_FN_HOST_ACC float* data() { return m_data; }

    constexpr ALPAKA_FN_HOST_ACC const auto& operator[](int i) const { return m_data[i]; }
    constexpr ALPAKA_FN_HOST_ACC auto& operator[](int i) { return m_data[i]; }
    ALPAKA_FN_HOST_ACC float range(int i) const { return m_data[i].max - m_data[i].min; }
  };

}  // namespace clue::internal
