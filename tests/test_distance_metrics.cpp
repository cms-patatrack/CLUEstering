
#include "CLUEstering/CLUEstering.hpp"
#include <cmath>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test euclidian metric") {
  auto metric = clue::metrics::Euclidean<2>();
  std::array<float, 3> point1{1.f, 2.f, 0.f};
  std::array<float, 3> point2{4.f, 6.f, 0.f};
  CHECK(metric(point1, point2) == doctest::Approx(5.f));
}

TEST_CASE("Test weighted euclidian metric") {
  SUBCASE("Array constructor") {
    auto metric_weights = std::array<float, 2>{1.f, 2.f};
    auto metric = clue::metrics::WeightedEuclidean<2>(metric_weights);
    std::array<float, 3> point1{1.f, 2.f, 0.f};
    std::array<float, 3> point2{4.f, 6.f, 0.f};
    CHECK(metric(point1, point2) == doctest::Approx(std::sqrt(41.f)));
  }

  SUBCASE("Variadic values constructor") {
    auto metric = clue::metrics::WeightedEuclidean<2>(1.f, 2.f);
    std::array<float, 3> point1{1.f, 2.f, 0.f};
    std::array<float, 3> point2{4.f, 6.f, 0.f};
    CHECK(metric(point1, point2) == doctest::Approx(std::sqrt(41.f)));
  }
}

TEST_CASE("Test periodic euclidian metric") {}

TEST_CASE("Test manhattan metric") {
  auto metric = clue::metrics::Manhattan<2>();
  std::array<float, 3> point1{-1.f, 3.f, 0.f};
  std::array<float, 3> point2{0.f, 6.f, 0.f};
  CHECK(metric(point1, point2) == doctest::Approx(4.f));
}

TEST_CASE("Test chebyshev metric") {
  auto metric = clue::metrics::Chebyshev<2>();
  std::array<float, 3> point1{-1.f, 4.f, 0.f};
  std::array<float, 3> point2{0.f, 6.f, 0.f};
  CHECK(metric(point1, point2) == doctest::Approx(2.f));
}

TEST_CASE("Test weighted chebyshev metric") {
  SUBCASE("Array constructor") {
    auto metric_weights = std::array<float, 2>{1.f, 2.f};
    auto metric = clue::metrics::WeightedChebyshev<2>(metric_weights);
    std::array<float, 3> point1{-1.f, 4.f, 0.f};
    std::array<float, 3> point2{0.f, 6.f, 0.f};
    CHECK(metric(point1, point2) == doctest::Approx(4.f));
  }

  SUBCASE("Variadic values constructor") {
    auto metric = clue::metrics::WeightedChebyshev<2>(1.f, 2.f);
    std::array<float, 3> point1{-1.f, 4.f, 0.f};
    std::array<float, 3> point2{0.f, 6.f, 0.f};
    CHECK(metric(point1, point2) == doctest::Approx(4.f));
  }
}
