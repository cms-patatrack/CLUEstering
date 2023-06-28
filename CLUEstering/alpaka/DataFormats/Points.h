#ifndef points_h
#define points_h

#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

template <uint8_t Ndim>
struct Points {
  Points() = default;

  std::vector<std::array<float, Ndim>> coordinates_;
  std::vector<float> weight;
  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<int> clusterIndex;
  std::vector<int> isSeed;
};

#endif
