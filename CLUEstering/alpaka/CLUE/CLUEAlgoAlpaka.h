#ifndef clustering_h
#define clustering_h

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>
#include <utility>

#include "ConvolutionalKernel.h"
#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "../DataFormats/alpaka/TilesAlpaka.h"
#include "../DataFormats/Math/DeltaPhi.h"

/* struct domain_t { */
/*   float min = -std::numeric_limits<float>::max(); */
/*   float max = std::numeric_limits<float>::max(); */
/* }; */

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TAcc, uint8_t Ndim>
  class CLUEAlgoAlpaka {
  public:
	CLUEAlgoAlpaka() = delete;
	explicit CLUEAlgoAlpaka(float dc, float rhoc, float outlierDeltaFactor, int pPBin) {
	  dc_ = dc;
	  rhoc_ = rhoc;
	  outlierDeltaFactor_ = outlierDeltaFactor;
	  pointsPerTile_ = pPBin;
	  /* domains_ = std::move(domains); */
	}

	Points<Ndim> m_points_h;
	TilesAlpaka<TAcc, Ndim>* m_tiles;

	/* int calculateNTiles(int pointPerBin) { */
	/*   int ntiles{points_.n / pointPerBin}; */
	/*   try { */
	/* 	if (ntiles == 0) { */
	/* 	  throw 100; */
	/* 	} */
	/*   } catch (...) { */
	/* 	std::cout */
	/* 		<< "pointPerBin is set too high for you number of points. You must lower it in the clusterer constructor.\n"; */
	/*   } */
	/*   return ntiles; */
	/* } */

	/* std::array<float, Ndim> calculateTileSize(int NTiles, tiles<Ndim>& tiles_) { */
	/*   std::array<float, Ndim> tileSizes; */
	/*   int NperDim{static_cast<int>(std::pow(NTiles, 1.0 / Ndim))}; */

	/*   for (int i{}; i != Ndim; ++i) { */
	/* 	float tileSize; */
	/* 	float dimMax{*std::max_element(points_.coordinates_[i].begin(), points_.coordinates_[i].end())}; */
	/* 	float dimMin{*std::min_element(points_.coordinates_[i].begin(), points_.coordinates_[i].end())}; */
	/* 	tiles_.minMax[i] = {dimMin, dimMax}; */
	/* 	tileSize = (dimMax - dimMin) / NperDim; */

	/* 	tileSizes[i] = tileSize; */
	/*   } */
	/*   return tileSizes; */
	/* } */

	/* std::vector<std::vector<int>> makeClusters(const kernel& ker) { */
	/*   tiles<Ndim> Tiles; */
	/*   Tiles.nTiles = calculateNTiles(pointsPerTile_); */
	/*   Tiles.resizeTiles(); */
	/*   Tiles.tilesSize = calculateTileSize(Tiles.nTiles, Tiles); */

	/*   prepareDataStructures(Tiles); */
	/*   calculateLocalDensity(Tiles, ker); */
	/*   calculateDistanceToHigher(Tiles); */
	/*   findAndAssignClusters(); */

	/*   return {points_.clusterIndex, points_.isSeed}; */
	/* } */

  private:
	float dc_;    // cut-off distance in the calculation of local density
	float rhoc_;  // minimum density to promote a point as a seed or the maximum density to demote a point as an outlier
	float outlierDeltaFactor_;
	int pointsPerTile_;  // average number of points found in a tile

	// Array containing the domain extremes of every coordinate
	/* std::array<domain_t, Ndim> domains_; */

	/* void calculateDistanceToHigher(tiles<Ndim>& tiles) { */
	/*   float dm{outlierDeltaFactor_ * dc_}; */

	/*   // loop over all points */
	/*   for (int i{}; i < points_.n; ++i) { */
	/* 	// default values of delta and nearest higher for i */
	/* 	float delta_i{std::numeric_limits<float>::max()}; */
	/* 	int nearestHigher_i{-1};  // if this doesn't change, the point is either a seed or an outlier */
	/* 	float rho_i{points_.rho[i]}; */

	/* 	// get search box */
	/* 	std::array<std::vector<int>, Ndim> xjBins; */
	/* 	for (int j{}; j != Ndim; ++j) { */
	/* 	  xjBins[j] = tiles.getBinsFromRange(points_.coordinates_[j][i] - dm, points_.coordinates_[j][i] + dm, j); */

	/* 	  // Overflow */
	/* 	  if (points_.coordinates_[j][i] + dm > domains_[j].max) { */
	/* 		std::vector<int> overflowBins = std::move(tiles.getBinsFromRange(domains_[j].min, domains_[j].min + dm, j)); */
	/* 		xjBins[j].insert(xjBins[j].end(), overflowBins.begin(), overflowBins.end()); */
	/* 		// Underflow */
	/* 	  } else if (points_.coordinates_[j][i] - dm < domains_[j].min) { */
	/* 		std::vector<int> underflowBins = std::move(tiles.getBinsFromRange(domains_[j].max - dm, domains_[j].max, j)); */
	/* 		xjBins[j].insert(xjBins[j].end(), underflowBins.begin(), underflowBins.end()); */
	/* 	  } */
	/* 	} */
	/* 	std::vector<int> search_box; */
	/* 	tiles.searchBox(xjBins, search_box); */

	/* 	// loop over all bins in the search box */
	/* 	for (int binId : search_box) { */
	/* 	  // get the size of this bin */
	/* 	  size_t binSize{tiles[binId].size()}; */

	/* 	  // iterate inside this bin */
	/* 	  for (size_t binIter{}; binIter < binSize; ++binIter) { */
	/* 		int j{tiles[binId][binIter]}; */
	/* 		// query N'_{dm}(i) */
	/* 		bool foundHigher{(points_.rho[j] > rho_i)}; */
	/* 		// in the rare case where rho is the same, use detid */
	/* 		foundHigher = foundHigher || ((points_.rho[j] == rho_i) && (j > i)); */
	/* 		float dist_ij{distance(i, j)}; */
	/* 		if (foundHigher && dist_ij <= dm) {  // definition of N'_{dm}(i) */
	/* 		  // find the nearest point within N'_{dm}(i) */
	/* 		  if (dist_ij < delta_i) { */
	/* 			// update delta_i and nearestHigher_i */
	/* 			delta_i = dist_ij; */
	/* 			nearestHigher_i = j; */
	/* 		  } */
	/* 		} */
		  /* }  // end of interate inside this bin */
		/* }    // end of loop over bins in the search box */

		/* points_.delta[i] = delta_i; */
		/* points_.nearestHigher[i] = nearestHigher_i; */
	  /* }  // end of loop over points */
	/* } */

	/* void findAndAssignClusters() { */
	  /* int nClusters{}; */

	  /* // find cluster seeds and outlier */
	  /* std::vector<int> localStack;  // this vector will contain the indexes of all the seeds */
	  /* // loop over all points */
	  /* for (int i{}; i < points_.n; ++i) { */
		/* // initialize clusterIndex */
		/* points_.clusterIndex[i] = -1; */

		/* float deltai{points_.delta[i]}; */
		/* float rhoi{points_.rho[i]}; */

		/* // determine seed or outlier */
		/* bool isSeed{(deltai > dc_) && (rhoi >= rhoc_)}; */
		/* bool isOutlier{(deltai > outlierDeltaFactor_ * dc_) && (rhoi < rhoc_)}; */
		/* if (isSeed) { */
		  /* // set isSeed as 1 */
		  /* points_.isSeed[i] = 1; */
		  /* // set cluster id */
		  /* points_.clusterIndex[i] = nClusters; */
		  /* // increment number of clusters */
		  /* ++nClusters; */
		  /* // add seed into local stack */
		  /* localStack.push_back(i); */
		/* } else if (!isOutlier) { */
		  /* // register as follower at its nearest higher */
		  /* points_.followers[points_.nearestHigher[i]].push_back(i); */
		/* } */
	  /* } */

	  /* // expend clusters from seeds */
	  /* while (!localStack.empty()) { */
		/* int i{localStack.back()}; */
		/* auto& followers{points_.followers[i]}; */
		/* localStack.pop_back(); */

		/* // loop over followers */
		/* for (int j : followers) { */
		  /* // pass id from i to a i's follower */
		  /* points_.clusterIndex[j] = points_.clusterIndex[i]; */
		  /* // push this follower to localStack */
		  /* localStack.push_back(j); */
		/* } */
	  /* } */
	/* } */

	/* inline float distance(int i, int j) const { */
	  /* float qSum{};  // quadratic sum */
	  /* for (int k{}; k != Ndim; ++k) { */
		/* float delta_xk{ */
			/* deltaPhi(points_.coordinates_[k][i], points_.coordinates_[k][j], domains_[k].min, domains_[k].max)}; */
		/* qSum += std::pow(delta_xk, 2); */
	  /* } */

	  /* return std::sqrt(qSum); */
	/* } */
  };

} 	// namespace ALPAKA_ACCELERATOR_NAMESPACE  
#endif
