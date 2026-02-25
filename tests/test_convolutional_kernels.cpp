
#include "CLUEstering/core/ConvolutionalKernel.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("Test FlatKernel throwing conditions") {
  CHECK_THROWS(clue::FlatKernel(0.f));
  CHECK_THROWS(clue::FlatKernel(-1.f));
}

TEST_CASE("Test ExponentialKernel throwing conditions") {
  CHECK_THROWS(clue::ExponentialKernel(0.f, 1.f));
  CHECK_THROWS(clue::ExponentialKernel(1.f, 0.f));
  CHECK_THROWS(clue::ExponentialKernel(-1.f, 1.f));
  CHECK_THROWS(clue::ExponentialKernel(1.f, -1.f));
}

TEST_CASE("Test GaussianKernel throwing conditions") {
  CHECK_THROWS(clue::GaussianKernel(0.f, 1.f, 1.f));
  CHECK_THROWS(clue::GaussianKernel(1.f, 0.f, 1.f));
  CHECK_THROWS(clue::GaussianKernel(1.f, 1.f, 0.f));
  CHECK_THROWS(clue::GaussianKernel(-1.f, 1.f, 1.f));
  CHECK_THROWS(clue::GaussianKernel(1.f, -1.f, 1.f));
  CHECK_THROWS(clue::GaussianKernel(1.f, 1.f, -1.f));
}
