#ifndef Points_h
#define Points_h

#include<vector>
#include<stdint.h>

template <uint8_t Ndim>
struct Points {
  std::vector<std::vector<float>> coordinates_;
  std::vector<float> weight;
  
  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<int> clusterIndex;
  std::vector<std::vector<int>> followers;
  std::vector<int> isSeed;
  int n;

  void clear() {
    for(int i = 0; i != Ndim; ++i) {
      coordinates_[i].clear();
    }
    weight.clear();

    rho.clear();
    delta.clear();
    nearestHigher.clear();
    clusterIndex.clear();
    followers.clear();
    isSeed.clear();

    n = 0;
  }
};
#endif
