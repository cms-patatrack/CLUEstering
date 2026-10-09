
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/detail/get_cluster_properties.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  // Minimal wrapper adapting cub::CachingDeviceAllocator to the allocator interface
  struct AllocatorWrapper {
    cub::CachingDeviceAllocator* allocator;

    void* allocate(std::size_t bytes, [[maybe_unused]] std::size_t align) {
      void* ptr = nullptr;
      if (allocator->DeviceAllocate(&ptr, bytes) != cudaSuccess)
        throw std::runtime_error("DeviceAllocate failed");
      return ptr;
    }

    void deallocate(void* ptr) {
      if (allocator->DeviceFree(ptr) != cudaSuccess)
        throw std::runtime_error("DeviceFree failed");
    }
  };

  static_assert(clue::concepts::external_allocator<AllocatorWrapper>);

  auto cached_bytes(cub::CachingDeviceAllocator& allocator) {
    int device;
    cudaGetDevice(&device);
    return allocator.cached_bytes[device];
  }

}  // namespace

TEST_CASE("Test clustering with the cub caching allocator") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};

  clue::PointsHost<2> h_points_default = clue::read_csv<2, float>(queue, test_file_path);
  clue::Clusterer<2> algo_default(queue, dc, rhoc, outlier);
  algo_default.make_clusters(queue, h_points_default);
  alpaka::wait(queue);

  cub::CachingDeviceAllocator cub_allocator;
  AllocatorWrapper allocator{&cub_allocator};

  SUBCASE("Results match the default allocator") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points);
    alpaka::wait(queue);

    CHECK(std::ranges::equal(h_points.clusterIndexes(), h_points_default.clusterIndexes()));
  }

  SUBCASE("Results match the default allocator using device points") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    clue::PointsDevice<2> d_points(queue, h_points.size(), allocator);
    clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points, d_points);
    alpaka::wait(queue);

    CHECK(std::ranges::equal(h_points.clusterIndexes(), h_points_default.clusterIndexes()));
  }

  SUBCASE("Internal buffers are allocated through the cub allocator") {
    {
      clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
      clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
      algo.make_clusters(queue, h_points);
      alpaka::wait(queue);

      // the clusterer keeps its internal buffers alive between runs
      CHECK(cached_bytes(cub_allocator).live > 0);
    }
    alpaka::wait(queue);

    // once the clusterer is destroyed all the memory is returned to the cache
    const auto bytes = cached_bytes(cub_allocator);
    CHECK(bytes.live == 0);
    CHECK(bytes.free > 0);
  }

  SUBCASE("Cached memory is reused by a new clusterer") {
    {
      clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
      clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
      algo.make_clusters(queue, h_points);
      alpaka::wait(queue);
    }
    alpaka::wait(queue);
    const auto first_run = cached_bytes(cub_allocator);

    {
      clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
      clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
      algo.make_clusters(queue, h_points);
      alpaka::wait(queue);

      CHECK(std::ranges::equal(h_points.clusterIndexes(), h_points_default.clusterIndexes()));
    }
    alpaka::wait(queue);
    const auto second_run = cached_bytes(cub_allocator);

    // no new device memory was requested for the second run
    CHECK(second_run.live == 0);
    CHECK(second_run.free == first_run.free);
  }
}

TEST_CASE("Test batched clustering with the cub caching allocator") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  cub::CachingDeviceAllocator cub_allocator;
  AllocatorWrapper allocator{&cub_allocator};

  clue::PointsHost<2> h_points =
      clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/batched_data_1024.csv");
  clue::PointsDevice<2> d_points(queue, h_points.size(), allocator);

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);

  std::vector<uint32_t> event_sizes(10, 1024);
  algo.make_clusters(queue, h_points, d_points, event_sizes);
  alpaka::wait(queue);

  auto truth = clue::read_output<2, float>(
      queue, std::string(TEST_DATA_DIR) + "/truth_files/data_1024_truth.csv");
  auto truth_n_clusters = clue::detail::compute_nclusters(truth.clusterIndexes());
  auto n_clusters = clue::detail::compute_nclusters(h_points.clusterIndexes());
  CHECK(n_clusters == truth_n_clusters * 10);
}

#endif
