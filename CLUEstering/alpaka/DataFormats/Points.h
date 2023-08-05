#ifndef points_h
#define points_h

#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include "alpaka/PointsAlpaka.h"
#include "alpaka/AlpakaVecArray.h"

using cms::alpakatools::VecArray;

template <uint8_t Ndim>
struct Points {
  Points() = default;

  std::vector<VecArray<float, Ndim>> coords;
  std::vector<float> weight;
  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<int> clusterIndex;
  std::vector<int> isSeed;

  int n;
};

#endif
