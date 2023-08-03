
#include <alpaka/dev/DevCpu.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "CLUEAlgoAlpaka.h"
#include "CLUEAlpakaKernels.h"

#include <alpaka/alpaka.hpp>
#include <cstdio>

using ALPAKA_ACCELERATOR_NAMESPACE::CLUEAlgoAlpaka;
using ALPAKA_ACCELERATOR_NAMESPACE::PointsAlpaka;

int main() {
  std::cout << __LINE__ << std::endl;
  using Dim = alpaka::DimInt<1u>;
  using Idx = uint32_t;

  using Acc = alpaka::AccCpuSerial<Dim, Idx>;

  using QueueProperty = alpaka::Blocking;
  using Queue = alpaka::Queue<Acc, QueueProperty>;
  using Device = alpaka::DevCpu;

  const float dc{1.f};
  const float rhoc{5.f};
  const float outlierDeltaFactor{1.5f};
  const int ppbin{10};

  auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);

  Queue queue_(dev_acc);

  using Vec = alpaka::Vec<Dim, Idx>;
  const Vec elementsPerThread(Vec::all(static_cast<Idx>(1)));
  const Vec threadsPerGrid(Vec::all(static_cast<Idx>(1)));
  using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
  WorkDiv const work_div = alpaka::getValidWorkDiv<Acc>(
      dev_acc, threadsPerGrid, elementsPerThread, false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

  CLUEAlgoAlpaka<Acc, 2> clueAlgo(dc, rhoc, outlierDeltaFactor, ppbin, queue_);

  Points<2> points_host;
  // Read points
  std::ifstream file_stream("./test_data.csv");
  std::string val;
  getline(file_stream, val);
  VecArray<float, 2> arr;
  float weight;

  int n_points;
  while (getline(file_stream, val, ',')) {
    arr[0] = std::stof(val);
    getline(file_stream, val, ',');
    arr[1] = std::stof(val);
    getline(file_stream, val);
    weight = std::stof(val);

    points_host.coords.push_back(arr);
    points_host.weight.push_back(weight);
    ++n_points;
  }
  std::cout << __LINE__ << std::endl;
  PointsAlpaka<2> points_dev(queue_, n_points);
  std::cout << __LINE__ << std::endl;

  auto tiles = clueAlgo.m_tiles;
  clueAlgo.make_clusters(points_host, points_dev, queue_);
}
