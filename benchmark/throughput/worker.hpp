
#pragma once

#include <array>
#include <cstddef>

constexpr std::size_t NDIM = 3;
constexpr int BLOCKSIZE = 512;

struct EventView {
  std::array<float*, NDIM> coords;
  float* weights;
  int* clusterIndexes;
  int n_points;
};

namespace serial {
  struct WorkerState;
  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points);
  void destroyWorker(WorkerState*);
  void processEvent(WorkerState*, const EventView&);
}  // namespace serial

namespace cuda {
  struct WorkerState;
  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points);
  void destroyWorker(WorkerState*);
  void processEvent(WorkerState*, const EventView&);
}  // namespace cuda
