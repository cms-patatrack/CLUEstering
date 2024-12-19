
#pragma once

#include "alpaka/AlpakaVecArray.h"
#include <vector>

using clue::VecArray;

template <uint8_t Ndim>
struct Points {
  Points() = default;
  Points(const std::vector<VecArray<float, Ndim>>& coords,
         const std::vector<float>& weight)
      : m_coords{coords}, m_weight{weight}, n{weight.size()} {
    m_rho.resize(n);
    m_delta.resize(n);
    m_nearestHigher.resize(n);
    m_clusterIndex.resize(n);
    m_isSeed.resize(n);
  }
  Points(const std::vector<std::vector<float>>& coords, const std::vector<float>& weight)
      : m_weight{weight}, n{weight.size()} {
    for (const auto& x : coords) {
      VecArray<float, Ndim> temp_vecarray;
      for (auto value : x) {
        temp_vecarray.push_back_unsafe(value);
      }
      m_coords.push_back(temp_vecarray);
    }

    m_rho.resize(n);
    m_delta.resize(n);
    m_nearestHigher.resize(n);
    m_clusterIndex.resize(n);
    m_isSeed.resize(n);
  }

  std::vector<VecArray<float, Ndim>> m_coords;
  std::vector<float> m_weight;
  std::vector<float> m_rho;
  std::vector<float> m_delta;
  std::vector<int> m_nearestHigher;
  std::vector<int> m_clusterIndex;
  std::vector<int> m_isSeed;

  size_t n;
};
