
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"

constexpr std::size_t NDIM = 3;
constexpr int BLOCKSIZE = 512;

namespace serial {
  struct WorkerState;
  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points);
  void destroyWorker(WorkerState*);
  void processEvent(WorkerState*, clue::PointsHost<NDIM>&);
}

namespace cuda {
  struct WorkerState;
  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points);
  void destroyWorker(WorkerState*);
  void processEvent(WorkerState*, clue::PointsHost<NDIM>&);
}
