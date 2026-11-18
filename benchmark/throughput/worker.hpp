
#pragma once

#include <array>
#include <cstddef>

constexpr std::size_t NDIM = 3;
constexpr int BLOCKSIZE = 512;

// Backend-agnostic view of an event's host memory. clue::PointsHost lives in a
// backend-specific inline namespace (clue::backend_serial, clue::backend_cuda, ...),
// so it can't cross the boundary between libraries built for different backends.
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
