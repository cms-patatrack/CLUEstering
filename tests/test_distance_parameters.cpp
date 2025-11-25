
#include "CLUEstering/CLUEstering.hpp"

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test distance parameter constructor") {
  SUBCASE("Construct one-dimensional distance parameter") {
    auto dist = clue::DistanceParameter<1>(2.5f);
    CHECK(dist[0] == 2.5f);
    auto dist2 = clue::DistanceParameter<1>(std::array<float, 1>{3.5f});
    CHECK(dist2[0] == 3.5f);
  }
  SUBCASE("Construct two-dimensional distance parameter") {
    auto dist = clue::DistanceParameter<2>(2.5f);
    CHECK(dist[0] == 2.5f);
    CHECK(dist[1] == 2.5f);
    auto dist2 = clue::DistanceParameter<2>(std::array<float, 2>{3.5f, 4.5f});
    CHECK(dist2[0] == 3.5f);
    CHECK(dist2[1] == 4.5f);
    auto dist3 = clue::DistanceParameter<2>(1.5f, 2.5f);
    CHECK(dist3[0] == 1.5f);
    CHECK(dist3[1] == 2.5f);
  }
  SUBCASE("Construct three-dimensional distance parameter") {
    auto dist = clue::DistanceParameter<3>(2.5f);
    CHECK(dist[0] == 2.5f);
    CHECK(dist[1] == 2.5f);
    CHECK(dist[2] == 2.5f);
    auto dist2 = clue::DistanceParameter<3>(std::array<float, 3>{3.5f, 4.5f, 5.5f});
    CHECK(dist2[0] == 3.5f);
    CHECK(dist2[1] == 4.5f);
    CHECK(dist2[2] == 5.5f);
    auto dist3 = clue::DistanceParameter<3>(1.5f, 2.5f, 3.5f);
    CHECK(dist3[0] == 1.5f);
    CHECK(dist3[1] == 2.5f);
    CHECK(dist3[2] == 3.5f);
  }
}

namespace {

  void foo1D(const clue::DistanceParameter<1>&) { CHECK(true); }
  void foo2D(const clue::DistanceParameter<2>&) { CHECK(true); }
  void foo3D(const clue::DistanceParameter<3>&) { CHECK(true); }

}  // namespace

TEST_CASE("Test distance parameter implicit conversion") {
  SUBCASE("Implicit conversion to one-dimensional distance parameter") {
    foo1D(2.5f);
    foo1D(std::array<float, 1>{3.5f});
  }
  SUBCASE("Implicit conversion to two-dimensional distance parameter") {
    foo2D(2.5f);
    foo2D(std::array<float, 2>{3.5f, 4.5f});
  }
  SUBCASE("Implicit conversion to three-dimensional distance parameter") {
    foo3D(2.5f);
    foo3D(std::array<float, 3>{3.5f, 4.5f, 5.5f});
  }
}

TEST_CASE("Test distance parameter comparison operators") {
  SUBCASE("Test one-dimensional distance parameter comparisons") {
    auto dist = clue::DistanceParameter<1>(2.5f);
    CHECK(dist < 3.0f);
    CHECK(dist <= 2.5f);
    CHECK(std::array{2.f} <= dist);
    CHECK_FALSE(std::array{2.f} > dist);
    CHECK(std::array{3.f} > dist);
    CHECK_FALSE(std::array{3.f} <= dist);
  }
  SUBCASE("Test two-dimensional distance parameter comparisons") {
    auto dist = clue::DistanceParameter<2>(2.5f);
    CHECK(dist < 3.0f);
    CHECK(dist <= 2.5f);
    CHECK(std::array{2.f, 2.f} <= dist);
    CHECK_FALSE(std::array{2.f, 2.6f} <= dist);
    CHECK_FALSE(std::array{2.6f, 2.4f} <= dist);
    CHECK(std::array{3.f, 2.4f} > dist);
    CHECK(std::array{2.4f, 3.f} > dist);
    CHECK(std::array{3.f, 3.f} > dist);
    CHECK_FALSE(std::array{2.5f, 2.5f} > dist);
    CHECK_FALSE(std::array{2.4f, 2.4f} > dist);

    dist = clue::DistanceParameter<2>(2.5f, 3.5f);
    CHECK(dist < 3.0f);
    CHECK(dist <= 2.5f);
    CHECK(std::array{2.f, 2.f} <= dist);
    CHECK_FALSE(std::array{2.f, 3.6f} <= dist);
    CHECK_FALSE(std::array{2.6f, 3.4f} <= dist);
    CHECK(std::array{3.f, 3.f} > dist);
    CHECK(std::array{2.4f, 3.6f} > dist);
    CHECK(std::array{2.6f, 3.4f} > dist);
  }
  SUBCASE("Test three-dimensional distance parameter comparisons") {
    auto dist = clue::DistanceParameter<3>(2.5f);
    CHECK(dist < 3.0f);
    CHECK(dist <= 2.5f);
    CHECK(std::array{2.0f, 2.0f, 2.0f} <= dist);
    CHECK_FALSE(std::array{2.6f, 2.0f, 2.0f} <= dist);
    CHECK_FALSE(std::array{2.0f, 2.6f, 2.0f} <= dist);
    CHECK_FALSE(std::array{2.0f, 2.0f, 2.6f} <= dist);
    CHECK_FALSE(std::array{2.0f, 2.6f, 2.6f} <= dist);
    CHECK(std::array{3.0f, 3.0f, 3.0f} > dist);
    CHECK(std::array{2.6f, 2.0f, 2.0f} > dist);
    CHECK(std::array{2.0f, 2.6f, 2.0f} > dist);
    CHECK(std::array{2.0f, 2.0f, 2.6f} > dist);
    CHECK(std::array{2.0f, 2.6f, 2.6f} > dist);

    dist = clue::DistanceParameter<3>(2.5f, 3.5f, 4.5f);
    CHECK(dist < 3.0f);
    CHECK(dist <= 2.5f);
    CHECK(std::array{2.0f, 2.0f, 2.0f} <= dist);
    CHECK(std::array{2.0f, 3.0f, 4.0f} <= dist);
    CHECK_FALSE(std::array{2.6f, 2.0f, 2.0f} <= dist);
    CHECK_FALSE(std::array{2.0f, 3.6f, 2.0f} <= dist);
    CHECK_FALSE(std::array{2.0f, 3.0f, 4.6f} <= dist);
    CHECK(std::array{2.6f, 3.0f, 4.0f} > dist);
    CHECK(std::array{2.0f, 3.6f, 4.0f} > dist);
    CHECK(std::array{2.0f, 3.0f, 4.6f} > dist);
  }
}

TEST_CASE("Test constexpr correctness") {
  constexpr auto dist = clue::DistanceParameter<3>(1.0f);
  static_assert(dist[1] == 1.0f);
  static_assert(dist < 2.0f);
}
