
#include <fstream>
#include <iostream>

#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "CLUEAlgoAlpaka.h"
#include "CLUEAlpakaKernels.h"

#include <alpaka/alpaka.hpp>
#include <cstdio>

using ALPAKA_ACCELERATOR_NAMESPACE::CLUEAlgoAlpaka;
using ALPAKA_ACCELERATOR_NAMESPACE::KernelPrepareDataStructures;

int main() {
  using Dim = alpaka::DimInt<3>;
  using Idx = std::size_t;

  using Acc = alpaka::AccCpuSerial<Dim, Idx>;

  using QueueProperty = alpaka::Blocking;
  using Queue = alpaka::Queue<Acc, QueueProperty>;

  const float dc{1.f};
  const float rhoc{5.f};
  const float outlierDeltaFactor{1.5f};
  const int ppbin{10};

  auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);

  Queue queue_(dev_acc);

  using Vec = alpaka::Vec<Dim, Idx>;
  Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
  Vec const threadsPerGrid(Vec::all(static_cast<Idx>(8)));
  using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
  WorkDiv const work_div = alpaka::getValidWorkDiv<Acc>(
      dev_acc, threadsPerGrid, elementsPerThread, false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

  CLUEAlgoAlpaka<Acc, 2> clueAlgo(dc, rhoc, outlierDeltaFactor, ppbin);

  Points<2> points;
  // Read points
  std::ifstream file_stream("./test_data.csv");
  std::string val;
  getline(file_stream, val);
  std::array<float, 2> arr;
  float weight;

  int n_points;
  while (getline(file_stream, val, ',')) {
    arr[0] = std::stof(val);
    getline(file_stream, val, ',');
    arr[1] = std::stof(val);
    getline(file_stream, val);
    weight = std::stof(val);

    points.coordinates_.push_back(arr);
    points.weight.push_back(weight);
    ++n_points;
  }

  clueAlgo.m_points_h = points;
  auto tiles = clueAlgo.m_tiles;
  alpaka::exec<Acc>(queue_, work_div, KernelPrepareDataStructures{}, points, n_points, tiles);

  return 0;
}