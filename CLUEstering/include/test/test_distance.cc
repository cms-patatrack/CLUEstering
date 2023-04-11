#include <cmath>
#include <iostream>
#include "../Clustering.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

inline float distance(ClusteringAlgo<2> algo, int i, int j) {
    float qSum{};  // quadratic sum
    for (int k{}; k != 2; ++k) {
      float delta_xk{};
      if (algo.domains_[k].empty()) {
        delta_xk = algo.points_.coordinates_[k][i] - algo.points_.coordinates_[k][j];
      } else {
        delta_xk = deltaPhi(algo.points_.coordinates_[k][i], algo.points_.coordinates_[k][j], algo.domains_[k].min, algo.domains_[k].max);
      }
      qSum += std::pow(delta_xk, 2);
    }

    return std::sqrt(qSum);
}

TEST_CASE(" ") {
    domain_t emp;
    domain_t dom{-M_PI, M_PI};
    ClusteringAlgo<2> obj(1., 5., 1.5, 10, {emp, dom});
    obj.setPoints(2, {{1., 1., 1., 1., 1.},{0.9*M_PI, -0.9*M_PI, 0., M_PI, 0.5*M_PI}}, {1,1,1,1,1});

    CHECK(doctest::Approx(distance(obj, 0, 1)).epsilon(0.000001) == (0.2*M_PI));
    CHECK(doctest::Approx(distance(obj, 0, 2)).epsilon(0.000001) == (0.9*M_PI));
    CHECK(doctest::Approx(distance(obj, 0, 3)).epsilon(0.000001) == (0.1*M_PI));
    CHECK(doctest::Approx(distance(obj, 0, 4)).epsilon(0.000001) == (0.4*M_PI));
    CHECK(doctest::Approx(distance(obj, 1, 2)).epsilon(0.000001) == (0.9*M_PI));
    CHECK(doctest::Approx(distance(obj, 1, 3)).epsilon(0.000001) == (0.1*M_PI));
    CHECK(doctest::Approx(distance(obj, 1, 4)).epsilon(0.000001) == (0.6*M_PI));
    CHECK(doctest::Approx(distance(obj, 2, 3)).epsilon(0.000001) == (1.*M_PI));
    CHECK(doctest::Approx(distance(obj, 2, 4)).epsilon(0.000001) == (0.5*M_PI));
    CHECK(doctest::Approx(distance(obj, 3, 4)).epsilon(0.000001) == (0.5*M_PI));
}