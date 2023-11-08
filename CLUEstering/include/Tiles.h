//
// Header for the tiles class, which represents the tiles that the clustering phase space is divided into
//

#ifndef tiles_h
#define tiles_h

#include <vector>
#include <iostream>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdint.h>

template <uint8_t Ndim>
class tiles {
private:
  // The elements of the outer vector represent the tiles/bins
  // The elements of the inner vectors represent the ids of the points in the corresponding tile
  std::vector<std::vector<int>> tiles_;

public:
  tiles() = default;
  void resizeTiles() { tiles_.resize(nTiles); }

  int nTiles;
  std::array<float, Ndim> tilesSize;
  std::array<std::vector<float>, Ndim> minMax;

  int getBin(float coord_, int dim_) const;
  int getGlobalBin(std::vector<float> const& coords) const;
  int getGlobalBinByBin(std::array<int, Ndim> const& Bins) const;
  void fill(std::vector<float> const& coords, int i);
  void fill(std::vector<std::vector<float>> const& coordinates);
  template <int N_>
  void searchBox(std::array<std::vector<int>, Ndim> const& xjBins,
                 std::array<int, Ndim>& comb,
                 std::vector<int>& sBox);
  void searchBox(std::array<std::vector<int>, Ndim> const& xjBins, std::vector<int>& sBox);
  std::vector<int> getBinsFromRange(float x_min, float x_max, int dim_);
  void clear();

  std::vector<int>& operator[](int globalBinId) { return tiles_[globalBinId]; }
};

template <uint8_t Ndim>
int tiles<Ndim>::getBin(float coord_, int dim_) const {
  int coord_Bin{static_cast<int>((coord_ - minMax[dim_][0]) / tilesSize[dim_])};
  coord_Bin = std::min(coord_Bin, static_cast<int>(std::pow(nTiles, 1.0 / Ndim) - 1));
  coord_Bin = std::max(coord_Bin, 0);
  return coord_Bin;
}

template <uint8_t Ndim>
int tiles<Ndim>::getGlobalBin(std::vector<float> const& coords) const {
  int globalBin{getBin(coords[0], 0)};
  int nTilesPerDim{static_cast<int>(std::pow(nTiles, 1.0 / Ndim))};
  for (int i{1}; i != Ndim; ++i) {
    globalBin += nTilesPerDim * getBin(coords[i], i);
  }
  return globalBin;
}

template <uint8_t Ndim>
int tiles<Ndim>::getGlobalBinByBin(std::array<int, Ndim> const& Bins) const {
  int globalBin{Bins[0]};
  int nTilesPerDim{static_cast<int>(std::pow(nTiles, 1.0 / Ndim))};
  for (int i{1}; i != Ndim; ++i) {
    globalBin += nTilesPerDim * Bins[i];
  }
  return globalBin;
}

template <uint8_t Ndim>
void tiles<Ndim>::fill(std::vector<float> const& coords, int i) {
  tiles_[getGlobalBin(coords)].push_back(i);
}

template <uint8_t Ndim>
void tiles<Ndim>::fill(std::vector<std::vector<float>> const& coordinates) {
  auto cellsSize = coordinates[0].size();
  for (int i{}; i < cellsSize; ++i) {
    std::vector<float> bin_coords;
    for (int j{}; j != Ndim; ++j) {
      bin_coords.push_back(coordinates[j][i]);
    }
    tiles_[getGlobalBin(bin_coords)].push_back(i);
  }
}

template <uint8_t Ndim>
template <int N_>
void tiles<Ndim>::searchBox(std::array<std::vector<int>, Ndim> const& xjBins,
                            std::array<int, Ndim>& comb,
                            std::vector<int>& sBox) {
  if constexpr (N_ == Ndim) {
    sBox.push_back(getGlobalBinByBin(comb));
  } else {
    for (int x : xjBins[N_]) {
      comb[N_] = x;
      searchBox<N_ + 1>(xjBins, comb, sBox);
    }
  }
}

template <uint8_t Ndim>
void tiles<Ndim>::searchBox(std::array<std::vector<int>, Ndim> const& xjBins, std::vector<int>& sBox) {
  for (int x : xjBins[0]) {
    std::array<int, Ndim> comb{x};
    searchBox<1>(xjBins, comb, sBox);
  }
}

template <uint8_t Ndim>
std::vector<int> tiles<Ndim>::getBinsFromRange(float x_min, float x_max, int dim_) {
  int minBin{getBin(x_min, dim_)};
  int maxBin{getBin(x_max, dim_)};

  std::vector<int> listOfBins(maxBin - minBin + 1);
  std::iota(listOfBins.begin(), listOfBins.end(), minBin);
  return listOfBins;
}

template <uint8_t Ndim>
void tiles<Ndim>::clear() {
  for (auto& t : tiles_) {
    t.clear();
  }
}

#endif
