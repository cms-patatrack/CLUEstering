#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include "clustering.h"

std::vector<std::vector<int>> run1(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<1> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run2(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<2> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}
			
std::vector<std::vector<int>> run3(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<3> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run4(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<4> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run5(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates,	std::vector<float> const& weight) {
	ClusteringAlgo<5> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run6(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<6> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run7(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<7> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run8(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<8> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run9(float dc, float rhoc, float outlier, int pPBin,  
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<9> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> run10(float dc, float rhoc, float outlier, int pPBin, 
		std::vector<std::vector<float>> const& coordinates, std::vector<float> const& weight) {
	ClusteringAlgo<10> algo(dc,rhoc,outlier,pPBin);
	algo.setPoints(coordinates[0].size(), coordinates, weight);

	return algo.makeClusters();
}

std::vector<std::vector<int>> mainRun(float dc, float rhoc, float outlier, int pPBin, 
            std::vector<std::vector<float>> const& coords, std::vector<float> const& weight, int Ndim) {
    // Running the clustering algorithm //
   	if (Ndim == 1) {
		return run1(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 2) {
		return run2(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 3) {
		return run3(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 4) {
		return run4(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 5) {
		return run5(dc,rhoc,outlier,pPBin,coords,weight);
		}
   	if (Ndim == 6) {
		return run6(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 7) {
		return run7(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 8) {
		return run8(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 9) {
		return run9(dc,rhoc,outlier,pPBin,coords,weight);
		} 
   	if (Ndim == 10) {
		return run10(dc,rhoc,outlier,pPBin,coords,weight);
		} 
}
