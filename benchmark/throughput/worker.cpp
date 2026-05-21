
#include "worker.hpp"
#include "defines.hpp"

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

#include <alpaka/alpaka.hpp>

namespace backend {

  using Acc = ALPAKA_BACKEND::Acc1D;
  using Device = ALPAKA_BACKEND::Device;
  using Queue = ALPAKA_BACKEND::Queue;

  struct WorkerState {
    Device device;
    Queue queue;
    clue::Clusterer<NDIM> clusterer;
    clue::PointsDevice<NDIM> d_points;

    WorkerState(float dc, float rhoc, float outlier, int n_points)
        : device(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u)),
          queue(device),
          clusterer(queue, dc, rhoc, outlier),
          d_points(queue, n_points) {}
  };

  WorkerState* createWorker(float dc, float rhoc, float outlier, int n_points) {
    return new WorkerState(dc, rhoc, outlier, n_points);
  }

  void destroyWorker(WorkerState* w) { delete w; }

  void processEvent(WorkerState* w, clue::PointsHost<NDIM>& h_points) {
    w->clusterer.make_clusters(w->queue, h_points, w->d_points);
    alpaka::wait(w->queue);
  }

}  // namespace backend
