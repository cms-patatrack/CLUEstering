
#include "Points.h"

#include <random>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test implementation of Points SoA") {
  const auto n = 1 << 10;
  const auto ndim = 2;
  std::vector<float> floatBuffer(3 * n);
  std::vector<int> intBuffer(2 * n);
  PointsSoA<2> points(floatBuffer.data(), intBuffer.data(), PointShape<2>{n});

  SUBCASE("Check that the content of the SoA is the same as the buffer") {
    CHECK(std::equal(floatBuffer.begin(), floatBuffer.end(), points.coords()));
    CHECK(std::equal(intBuffer.begin(), intBuffer.end(), points.clusterIndexes()));
  }
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> distfloat(0.f, 1.f);
  std::uniform_real_distribution<float> distint(0, n);
  SUBCASE("Update point data through view and compare with buffer") {
    auto view = points.view();
    // set the first coordinate
    std::generate(
        view->coords, view->coords + n, [&]() -> float { return distfloat(gen); });
    // set the second coordinate
    std::generate(view->coords + n, view->coords + ndim * n, [&]() -> float {
      return distfloat(gen);
    });
    // set the weights (third row)
    std::generate(view->coords + ndim * n, view->coords + (ndim + 1) * n, [&]() -> float {
      return distfloat(gen);
    });
    // set the two integer rows
    std::generate(view->clusterIndexes, view->clusterIndexes + 2 * n, [&]() -> int {
      return distint(gen);
    });

    // check content
    CHECK(std::equal(floatBuffer.begin(), floatBuffer.end(), points.coords()));
    CHECK(std::equal(floatBuffer.begin(), floatBuffer.end(), view->coords));
    CHECK(std::equal(intBuffer.begin(), intBuffer.end(), points.clusterIndexes()));
    CHECK(std::equal(intBuffer.begin(), intBuffer.end(), view->clusterIndexes));
  }
}
