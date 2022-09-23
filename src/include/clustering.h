#ifndef CLUEAlgo_h
#define CLUEAlgo_h

// C/C++ headers
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "tiles.h"
#include "point.h"

template <uint8_t Ndim>
class ClusteringAlgo{
public:
  ClusteringAlgo(float dc, float rhoc, float outlierDeltaFactor, int pPBin) {
    dc_ = dc; 
    rhoc_ = rhoc;
    outlierDeltaFactor_ = outlierDeltaFactor;
    pointsPerTile_ = pPBin;
  }
  ~ClusteringAlgo(){} 
    
  // public variables
  float dc_;  // cut-off distance in the calculation of local density
  float rhoc_;  // minimum density to promote a point as a seed or the maximum density to demote a point as an outlier
  float outlierDeltaFactor_;
  int pointsPerTile_; // average number of points found in a tile
    
  Points<Ndim> points_;
  
  bool setPoints(int n, std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	//points_.clear();
    // input variables
    for(int i = 0; i < n; ++i) {
		for(int j = 0; j != Ndim; ++j) {
			points_.coordinates_.push_back({});
            points_.coordinates_[j].push_back(coordinates[j][i]);
        }
	    points_.weight.push_back(weight[i]);
    }

    points_.n = points_.coordinates_[0].size();
    if(points_.n == 0)
      return 1;

    // result variables
    points_.rho.resize(points_.n,0);
    points_.delta.resize(points_.n,std::numeric_limits<float>::max());
    points_.nearestHigher.resize(points_.n,-1);
    points_.followers.resize(points_.n);
    points_.clusterIndex.resize(points_.n,-1);
    points_.isSeed.resize(points_.n,0);
    return 0;
  }

  void clearPoints(){ points_.clear(); }

  int calculateNTiles(int pointPerBin) {
    return points_.n/pointPerBin;
  }

  std::array<float,Ndim> calculateTileSize(int NTiles, tiles<Ndim>& tiles_) {
    std::array<float,Ndim> tileSizes;
    int NperDim = std::pow(NTiles,1.0/Ndim);

    for(int i = 0; i != Ndim; ++i) {
      float tileSize;
      float dimMax = *std::max_element(points_.coordinates_[i].begin(),points_.coordinates_[i].end());
      float dimMin = *std::min_element(points_.coordinates_[i].begin(),points_.coordinates_[i].end());
      tiles_.minMax[i] = {dimMin,dimMax};
      tileSize = (dimMax-dimMin)/NperDim;
      
      tileSizes[i] = tileSize;
    }

    return tileSizes;
  }

  std::vector<std::vector<int>> makeClusters() {
	tiles<Ndim> Tiles;
    Tiles.nTiles = calculateNTiles(pointsPerTile_);
    Tiles.resizeTiles();
  	Tiles.tilesSize = calculateTileSize(Tiles.nTiles, Tiles);

    // start clustering
    auto start = std::chrono::high_resolution_clock::now();
    prepareDataStructures(Tiles);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "--- prepareDataStructures:     " << elapsed.count() *1000 << " ms\n";
    
    start = std::chrono::high_resolution_clock::now();
    calculateLocalDensity(Tiles);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "--- calculateLocalDensity:     " << elapsed.count() *1000 << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    calculateDistanceToHigher(Tiles);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "--- calculateDistanceToHigher: " << elapsed.count() *1000 << " ms\n";

    findAndAssignClusters();

	return {points_.clusterIndex,points_.isSeed};
  }

  template <int N_>
  void for_recursion(std::vector<int> &base_vector,  std::vector<int> &dim_min, std::vector<int> &dim_max, tiles<Ndim>& lt_, int point_id) {
    if constexpr (N_ == 0) {
      int binId = lt_.getGlobalBinByBin(base_vector);
      // get the size of this bin
      int binSize = lt_[binId].size();
      
      // iterate inside this bin
      for (int binIter = 0; binIter < binSize; ++binIter) {
        int j = lt_[binId][binIter];
        // query N_{dc_}(i)
        float dist_ij = distance(point_id, j);

        if(dist_ij <= dc_) {
          // sum weights within N_{dc_}(i)
          points_.rho[point_id] += (point_id == j ? 1.f : 0.5f) * points_.weight[j];
        }
      } // end of interate inside this bin
      return;
    } else {
		 for(int i = dim_min[dim_min.size() - N_]; i <= dim_max[dim_max.size() - N_]; ++i) {
			  base_vector[base_vector.size() - N_] = i;
			  for_recursion<N_-1>(base_vector, dim_min, dim_max, lt_, point_id);
		 }
	 }
  }

  template <int N_>
  void for_recursion_DistanceToHigher(std::vector<int> &base_vector,  std::vector<int> &dim_min, std::vector<int> &dim_max, 
    tiles<Ndim>& lt_, float rho_i, float& delta_i, int& nearestHigher_i, int point_id) {
      if constexpr (N_ == 0) {
        float dm = outlierDeltaFactor_ * dc_;

        int binId = lt_.getGlobalBinByBin(base_vector);
        // get the size of this bin
        int binSize = lt_[binId].size();
        
        // iterate inside this bin
        for (int binIter = 0; binIter < binSize; ++binIter) {
          int j = lt_[binId][binIter]; 
          // query N'_{dm}(i)
          bool foundHigher = (points_.rho[j] > rho_i);
          // in the rare case where rho is the same, use detid
          foundHigher = foundHigher || ((points_.rho[j] == rho_i) && (j > point_id) );
          float dist_ij = distance(point_id, j);
          if(foundHigher && dist_ij <= dm) { // definition of N'_{dm}(i)
            // find the nearest point within N'_{dm}(i)
            if (dist_ij < delta_i) {
              // update delta_i and nearestHigher_i
              delta_i = dist_ij;
              nearestHigher_i = j;
            }
          }
        } // end of interate inside this bin

        return;
      } else {
			for(int i = dim_min[dim_min.size() - N_]; i <= dim_max[dim_max.size() - N_]; ++i){
				 base_vector[base_vector.size() - N_] = i;
				 for_recursion_DistanceToHigher<N_-1>(base_vector, dim_min, dim_max, lt_, rho_i, delta_i, nearestHigher_i, point_id);
			}
		}
  }

private:
  // private member methods
  void prepareDataStructures(tiles<Ndim>& tiles) {
	for (int i = 0; i < points_.n; ++i){
      // push index of points into tiles
      std::vector<float> coords;
      for(int j = 0; j != Ndim; ++j) {
        coords.push_back(points_.coordinates_[j][i]);
      }
      tiles.fill(coords, i);
    }
  }

  void calculateLocalDensity(tiles<Ndim>& tiles) {
    // loop over all points
    for(int i = 0; i < points_.n; ++i) {
      // get search box
      std::array<std::vector<float>,Ndim> minMax;
      for(int j = 0; j != Ndim; ++j) {
        std::vector<float> partial_minMax{points_.coordinates_[j][i]-dc_,points_.coordinates_[j][i]+dc_};
        minMax[j] = partial_minMax;
      }
      std::array<int,2*Ndim> search_box = tiles.searchBox(minMax);

      // loop over bins in the search box(binIter_f - binIter_i)
      std::vector<int> binVec(Ndim);
      std::vector<int> dimMin;
      std::vector<int> dimMax;
      for(int j = 0; j != (int)(search_box.size()); ++j) {
        if(j%2 == 0) {
          dimMin.push_back(search_box[j]);
        } else {
          dimMax.push_back(search_box[j]);
        }
      }

      //for_recursion<Ndim>(binVec,dimMin,dimMax,tiles,i);
		for_recursion<Ndim>(binVec,dimMin,dimMax,tiles,i);
    } // end of loop over points
  }

  void calculateDistanceToHigher(tiles<Ndim>& tiles) {
    float dm = outlierDeltaFactor_ * dc_;
    
    // loop over all points
    for(int i = 0; i < points_.n; ++i) {
      // default values of delta and nearest higher for i
      float delta_i = std::numeric_limits<float>::max();
      int nearestHigher_i = -1; // if this doesn't change, the point is either a seed or an outlier
      float rho_i = points_.rho[i];

      // get search box
      std::array<std::vector<float>,Ndim> minMax;
      for(int j = 0; j != Ndim; ++j) {
        std::vector<float> partial_minMax{points_.coordinates_[j][i]-dm,points_.coordinates_[j][i]+dm};
        minMax[j] = partial_minMax;
      }
      std::array<int,2*Ndim> search_box = tiles.searchBox(minMax);

      // loop over all bins in the search box

      std::vector<int> binVec(Ndim);
      std::vector<int> dimMin;
      std::vector<int> dimMax;
      for(int j = 0; j != search_box.size(); ++j) {
        if(j%2 == 0) {
          dimMin.push_back(search_box[j]);
        } else {
          dimMax.push_back(search_box[j]);
        }
      }
      for_recursion_DistanceToHigher<Ndim>(binVec,dimMin,dimMax,tiles, rho_i, delta_i, nearestHigher_i, i);

      points_.delta[i] = delta_i;
      points_.nearestHigher[i] = nearestHigher_i;
    } // end of loop over points
  }

  void findAndAssignClusters() {
     auto start = std::chrono::high_resolution_clock::now();

    int nClusters = 0;

    // find cluster seeds and outlier
    std::vector<int> localStack;  // this vector will contain the indexes of all the seeds
    // loop over all points
    for(int i = 0; i < points_.n; ++i) {
      // initialize clusterIndex
      points_.clusterIndex[i] = -1;

      float deltai = points_.delta[i];
      float rhoi = points_.rho[i];

      // determine seed or outlier 
      bool isSeed = (deltai > dc_) && (rhoi >= rhoc_);
      bool isOutlier = (deltai > outlierDeltaFactor_ * dc_) && (rhoi < rhoc_);
      if (isSeed) {
	      // set isSeed as 1
	      points_.isSeed[i] = 1;
	      // set cluster id
	      points_.clusterIndex[i] = nClusters;
	      // increment number of clusters
	      ++nClusters;
	      // add seed into local stack
	      localStack.push_back(i);
      } else if (!isOutlier) {
	      // register as follower at its nearest higher
	      points_.followers[points_.nearestHigher[i]].push_back(i);
      }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "--- findSeedAndFollowers:      " << elapsed.count() *1000 << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    // expend clusters from seeds
    while (!localStack.empty()) {
      int i = localStack.back();
      auto& followers = points_.followers[i];
      localStack.pop_back();

      // loop over followers
      for(int j : followers){
        // pass id from i to a i's follower
        points_.clusterIndex[j] = points_.clusterIndex[i];
        // push this follower to localStack
        localStack.push_back(j);
      }
    }
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "--- assignClusters:            " << elapsed.count() *1000 << " ms\n";
  }

  inline float distance(int i, int j) const {
    float qSum = 0;   // quadratic sum
    for(int k = 0; k != Ndim; ++k) {
      qSum += std::pow(points_.coordinates_[k][i] - points_.coordinates_[k][j],2);
    }
    
    return std::sqrt(qSum);
  }
};

#endif
