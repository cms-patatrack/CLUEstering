
#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <std::size_t Ndim>
  class DistanceParameter {
  private:
    std::array<float, Ndim> m_parameters;

  public:
    constexpr DistanceParameter(float radius) : m_parameters{} {
      std::ranges::fill(m_parameters, radius);
    }
    constexpr DistanceParameter(std::array<float, Ndim> distances)
        : m_parameters{std::move(distances)} {}
    template <std::floating_point... TDistances>
    constexpr DistanceParameter(TDistances... distances) : m_parameters{distances...} {}

    constexpr const auto& operator[](std::size_t dim) const { return m_parameters[dim]; }
    constexpr auto& operator[](std::size_t dim) { return m_parameters[dim]; }

    constexpr bool operator<(float value) {
      return std::ranges::any_of(m_parameters, [=](auto x) -> bool { return x < value; });
    }
    constexpr bool operator<=(float value) {
      return std::ranges::any_of(m_parameters, [=](auto x) -> bool { return x <= value; });
    }

  private:
    // TODO: needed to avoid failed template substitution
    // to be removed when changing type of Ndim
    template <std::size_t N1, std::size_t N2>
      requires(N1 == N2)
    ALPAKA_FN_ACC friend bool operator<=(const std::array<float, N1>&,
                                         const DistanceParameter<N2>&);
    // TODO: needed to avoid failed template substitution
    // to be removed when changing type of Ndim
    template <std::size_t N1, std::size_t N2>
      requires(N1 == N2)
    ALPAKA_FN_ACC friend bool operator>(const std::array<float, N1>&, const DistanceParameter<N2>&);
  };

  template <std::size_t N1, std::size_t N2>
    requires(N1 == N2)
  ALPAKA_FN_ACC bool operator<=(std::array<float, N1>& lhs, const DistanceParameter<N2>& rhs) {
    return [&]<std::size_t... Ids>(std::index_sequence<Ids...>) -> bool {
      return ((lhs[Ids] <= rhs[Ids]) && ...);
    }(std::make_index_sequence<N1>{});
  }

  template <std::size_t N1, std::size_t N2>
    requires(N1 == N2)
  ALPAKA_FN_ACC bool operator>(std::array<float, N1>& lhs, const DistanceParameter<N2>& rhs) {
    return [&]<std::size_t... Ids>(std::index_sequence<Ids...>) -> bool {
      return ((lhs[Ids] > rhs[Ids]) && ...);
    }(std::make_index_sequence<N1>{});
  }

}  // namespace clue
