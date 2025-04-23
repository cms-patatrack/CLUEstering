
#include "CLUEstering.hpp"
#include "utility/read_csv.hpp"
#include "utility/validation.hpp"

#include <cmath>
#include <format>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

TEST_CASE("Test clustering on benchmarking datasets") {
  for (auto i = 10; i < 19; ++i) {
    auto coords = read_csv<float, 2>(std::format("../data/data_{}.csv", std::pow(2, i)));
    const uint32_t n_points = coords.size() / 3;
    std::vector<int> results(2 * n_points);

    const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
    Queue queue(dev_acc);

    clue::PointsHost<2> h_points(queue, n_points, coords, results);
    clue::PointsDevice<2, Device> d_points(queue, n_points);

    const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
    const int pPBin{128};
    CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue);

    const std::size_t block_size{256};
    algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
    auto clusters = std::span<const int>{results.data(), n_points};
    auto isSeed = std::span<const int>(results.data() + n_points, n_points);

    const auto truth_data = read_output<2>(
        std::format("../data/truth_files/data_{}_truth.csv", std::pow(2, i)));
    auto truth_ids = std::span<const int>{truth_data.data(), n_points};
    auto truth_isSeed = std::span<const int>{truth_data.data() + n_points, n_points};
    CHECK(clue::validate_results(clusters, truth_ids));
    CHECK(std::ranges::equal(truth_isSeed, isSeed));
  }
}

TEST_CASE("Test clustering on sissa") {
  auto coords = read_csv<float, 2>("../data/sissa.csv");
  const uint32_t n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<2> h_points(queue, n_points, coords, results);
  clue::PointsDevice<2, Device> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  const int pPBin{128};
  CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = std::span<const int>{results.data(), n_points};
  auto isSeed = std::span<const int>(results.data() + n_points, n_points);

  const auto truth_data = read_output<2>("../data/truth_files/sissa_1000_truth.csv");
  auto truth_ids = std::span<const int>{truth_data.data(), n_points};
  auto truth_isSeed = std::span<const int>{truth_data.data() + n_points, n_points};
  CHECK(clue::validate_results(clusters, truth_ids));
  CHECK(std::ranges::equal(truth_isSeed, isSeed));
}

TEST_CASE("Test clustering on toy detector dataset") {
  auto coords = read_csv<float, 2>("../data/toyDetector.csv");
  const uint32_t n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<2> h_points(queue, n_points, coords, results);
  clue::PointsDevice<2, Device> d_points(queue, n_points);

  const float dc{4.5f}, rhoc{2.5f}, outlier{4.5f};
  const int pPBin{128};
  CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = std::span<const int>{results.data(), n_points};
  auto isSeed = std::span<const int>(results.data() + n_points, n_points);

  const auto truth_data = read_output<2>("../data/truth_files/toy_det_1000_truth.csv");
  auto truth_ids = std::span<const int>{truth_data.data(), n_points};
  auto truth_isSeed = std::span<const int>{truth_data.data() + n_points, n_points};
  CHECK(clue::validate_results(clusters, truth_ids));
  CHECK(std::ranges::equal(truth_isSeed, isSeed));
}

TEST_CASE("Test clustering on blob dataset") {
  auto coords = read_csv<float, 3>("../data/blob.csv");
  const uint32_t n_points = coords.size() / 4;
  std::vector<int> results(2 * n_points);

  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  clue::PointsHost<3> h_points(queue, n_points, coords, results);
  clue::PointsDevice<3, Device> d_points(queue, n_points);

  const float dc{1.f}, rhoc{5.f}, outlier{2.f};
  const int pPBin{128};
  CLUEAlgoAlpaka<3> algo(dc, rhoc, outlier, pPBin, queue);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = std::span<const int>{results.data(), n_points};
  auto isSeed = std::span<const int>(results.data() + n_points, n_points);

  const auto truth_data = read_output<3>("../data/truth_files/blobs_truth.csv");
  auto truth_ids = std::span<const int>{truth_data.data(), n_points};
  auto truth_isSeed = std::span<const int>{truth_data.data() + n_points, n_points};
  CHECK(clue::validate_results(clusters, truth_ids));
  CHECK(std::ranges::equal(truth_isSeed, isSeed));
}
