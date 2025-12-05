
#pragma once

#include <alpaka/alpaka.hpp>
#include <array>
#include <cstddef>

namespace clue::internal {

  template <typename TMetric, std::size_t Ndim>
  struct MetricInterface {
  private:
    using Point = std::array<float, Ndim + 1>;

  public:
    ALPAKA_FN_HOST_ACC constexpr auto operator()(const Point& lhs, const Point& rhs) const {
      return static_cast<const TMetric*>(this)->distance(lhs, rhs);
    }
  };

}  // namespace clue::internal
