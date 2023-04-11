#include <cmath>
#include "../deltaPhi.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Testing deltaPhi in (0, 2pi) domain") {
    CHECK(deltaPhi(0., 3.*M_PI/2., 0., 2.*M_PI) == 0.5*M_PI);
    CHECK(deltaPhi(0., M_PI/2., 0., 2.*M_PI) == M_PI/2.);
    CHECK(deltaPhi(M_PI/2., 0., 0., 2.*M_PI) == M_PI/2.);
    CHECK(deltaPhi(0.5*M_PI, 0.6*M_PI, 0., 2.*M_PI) == 0.1*M_PI);
    CHECK(deltaPhi(0.1*M_PI, 1.9*M_PI, 0., 2.*M_PI) == 0.2*M_PI);
}

TEST_CASE("Testing deltaPhi in (-pi, pi) domain") {
    CHECK(deltaPhi(0., M_PI, -M_PI, M_PI) == M_PI);
    CHECK(deltaPhi(-M_PI, 0., -M_PI, M_PI) == M_PI);
    CHECK(deltaPhi(-M_PI, M_PI, -M_PI, M_PI) == 0.);
    CHECK(deltaPhi(-0.9*M_PI, 0.9*M_PI, -M_PI, M_PI) == 0.2*M_PI);
    CHECK(deltaPhi(-0.5*M_PI, M_PI, -M_PI, M_PI) == 0.5*M_PI);
    CHECK(deltaPhi(-0.5*M_PI, M_PI, -M_PI, M_PI) == deltaPhi(-M_PI, 0.5*M_PI, -M_PI, M_PI));
}

TEST_CASE("Testing deltaPhi in an asymmetric domain, (-0.5pi, 3pi)") {
    CHECK(deltaPhi(0., M_PI, -0.5*M_PI, 3*M_PI) == M_PI);
    CHECK(deltaPhi(0., 2*M_PI, -0.5*M_PI, 3*M_PI) == 1.5*M_PI);
    CHECK(deltaPhi(-0.5*M_PI, 3*M_PI, -0.5*M_PI, 3*M_PI) == 0.);
    CHECK(deltaPhi(-0.5*M_PI, 2.9*M_PI, -0.5*M_PI, 3*M_PI) == 0.1*M_PI);
    CHECK(deltaPhi(-0.1*M_PI, 0., -0.5*M_PI, 3*M_PI) == 0.1*M_PI);
}