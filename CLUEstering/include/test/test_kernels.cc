#include <cmath>
#include <iostream>
#include "../Clustering.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test the base kernel") {
  CHECK(kernel()(0., 0, 0) == 0);
  CHECK(kernel()(1.5, 0, 0) == 0);
  CHECK(kernel()(0., 1, 0) == 0);
  CHECK(kernel()(0., 0, 1) == 0);
  CHECK(kernel()(1.5, 0, 1) == 0);
}

TEST_CASE("Test the flat kernel kernel") {
  CHECK(flatKernel(.5)(0., 0, 0) == 1.);
  CHECK(flatKernel(.5)(1.5, 0, 0) == 1.);
  CHECK(flatKernel(.5)(0., 1, 0) == .5);
  CHECK(flatKernel(.5)(0., 0, 1) == .5);
  CHECK(flatKernel(.5)(1.5, 0, 1) == .5);
}

TEST_CASE("Test the gaussian kernel kernel") {
  CHECK(gaussianKernel(1.5, 1., 1.)(0., 0, 0) == 1.);
  CHECK(gaussianKernel(1.5, 1., 1.)(2., 0, 0) == 1.);
  CHECK(doctest::Approx(gaussianKernel(1.5, 1., 1.)(0., 1, 0)).epsilon(0.000001) == std::exp(-(1.5 * 1.5) / 2));
  CHECK(doctest::Approx(gaussianKernel(1.5, 1., 1.)(0., 0, 1)).epsilon(0.000001) == std::exp(-(1.5 * 1.5) / 2));
  CHECK(doctest::Approx(gaussianKernel(1.5, 1., 1.)(2., 0, 1)).epsilon(0.000001) == std::exp(-(0.5 * 0.5) / 2));
}

TEST_CASE("Test the exponential kernel") {
  CHECK(exponentialKernel(.5, 1.)(0., 0, 0) == 1.);
  CHECK(exponentialKernel(.5, 1.)(2., 0, 0) == 1.);
  CHECK(exponentialKernel(.5, 1.)(0., 1, 0) == 1.);
  CHECK(exponentialKernel(.5, 1.)(0., 0, 1) == 1.);
  CHECK(doctest::Approx(exponentialKernel(.5, 1.)(2., 0, 1)).epsilon(0.000001) == std::exp(-1.));
}
