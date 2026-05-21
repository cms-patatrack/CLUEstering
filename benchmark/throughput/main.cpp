
#include "worker.hpp"

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "utils/generation.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <alpaka/alpaka.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

constexpr auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;

using Event = clue::PointsHost<NDIM>;

auto runEvents(int nCpuThreads, int nGpuStreams, int nEvents, int nPoints, int nClusters) {
  const auto cpuDev = alpaka::getDevByIdx(alpaka::Platform<alpaka_serial_sync::Acc1D>{}, 0u);
  alpaka_serial_sync::Queue genQueue(cpuDev);

  std::vector<Event> events;
  events.reserve(nEvents);
  for (auto i = 0; i < nEvents; ++i) {
    events.emplace_back(clue::utils::generateRandomData<NDIM>(
        genQueue, nPoints, nClusters, std::make_pair(-100.f, 100.f), 1.f, 0.1f, i));
  }

  std::vector<serial::WorkerState*> cpuWorkers(nCpuThreads);
  for (auto& w : cpuWorkers)
    w = serial::createWorker(dc, rhoc, outlier, nPoints);

#ifdef CLUE_HAS_CUDA
  std::vector<cuda::WorkerState*> gpuWorkers(nGpuStreams);
  for (auto& w : gpuWorkers)
    w = cuda::createWorker(dc, rhoc, outlier, nPoints);
#else
  if (nGpuStreams > 0) {
    std::cerr << "GPU support not available in this build\n";
    return 0.;
  }
#endif

  std::atomic<int> eventCounter = 0;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread cpuThread([&] {
    if (nCpuThreads == 0)
      return;
    tbb::task_arena arena(nCpuThreads);
    arena.execute([&] {
      tbb::parallel_for(0, nCpuThreads, [&](int i) {
        while (true) {
          int id = eventCounter.fetch_add(1);
          if (id >= nEvents)
            return;
          serial::processEvent(cpuWorkers[i], events[id]);
        }
      });
    });
  });

#ifdef CLUE_HAS_CUDA
  std::thread gpuThread([&] {
    if (nGpuStreams == 0)
      return;
    tbb::task_arena arena(nGpuStreams);
    arena.execute([&] {
      tbb::parallel_for(0, nGpuStreams, [&](int i) {
        while (true) {
          int id = eventCounter.fetch_add(1);
          if (id >= nEvents)
            return;
          cuda::processEvent(gpuWorkers[i], events[id]);
        }
      });
    });
  });
#endif

  cpuThread.join();
#ifdef CLUE_HAS_CUDA
  gpuThread.join();
#endif

  auto end = std::chrono::high_resolution_clock::now();

  for (auto& w : cpuWorkers)
    serial::destroyWorker(w);
#ifdef CLUE_HAS_CUDA
  for (auto& w : gpuWorkers)
    cuda::destroyWorker(w);
#endif

  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return (1000. * nEvents) / duration;
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <nCpuThreads> <nGpuStreams> <nEvents> <nPoints> <nClusters>\n";
    return 1;
  }
  const auto nCpuThreads = std::stoi(argv[1]);
  const auto nGpuStreams = std::stoi(argv[2]);
  const auto nEvents = std::stoi(argv[3]);
  const auto nPoints = std::stoi(argv[4]);
  const auto nClusters = std::stoi(argv[5]);

  const auto throughput = runEvents(nCpuThreads, nGpuStreams, nEvents, nPoints, nClusters);
  std::cout << "Throughput = " << throughput << " events/s\n";
}
