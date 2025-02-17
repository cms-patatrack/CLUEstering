
#include "CLUEstering.hpp"
#include "utility/read_csv.hpp"
#include "utility/validation.hpp"
#include <alpaka/alpaka.hpp>

#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

TEST_CASE("Test clustering using externally defined Tiles") {
  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  auto coords = read_csv<float, 2>("./sissa.csv");
  const auto n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  PointsSoA<2> h_points(coords.data(), results.data(), PointShape<2>{n_points});
  PointsAlpaka<2> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  const int pPBin{128};

  // construct the tiles
  auto tiles = TilesAlpaka<2>(queue, n_points, pPBin);

  CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue, &tiles);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);

  auto truth = read_output<2>("./sissa_1000_truth.csv");

  CHECK(clue::validate_results(std::span{results.data(), n_points},
                               std::span{truth.data(), n_points}));
}

TEST_CASE("Test clustering using Tiles from allocated buffer") {
  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  auto coords = read_csv<float, 2>("./sissa.csv");
  const auto n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  PointsSoA<2> h_points(coords.data(), results.data(), PointShape<2>{n_points});
  PointsAlpaka<2> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  const int pPBin{128};

  // construct the tiles
  alignas(TilesAlpaka<2>) unsigned char buffer[sizeof(TilesAlpaka<2>)];
  auto tiles = new (buffer) TilesAlpaka<2>(queue, n_points, pPBin);

  CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue, tiles);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  tiles->~TilesAlpaka<2>();

  auto truth = read_output<2>("./sissa_1000_truth.csv");

  CHECK(clue::validate_results(std::span{results.data(), n_points},
                               std::span{truth.data(), n_points}));
}

TEST_CASE("Test clustering using Tiles from allocated large buffer") {
  const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(dev_acc);

  auto coords = read_csv<float, 2>("./sissa.csv");
  const auto n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  PointsSoA<2> h_points(coords.data(), results.data(), PointShape<2>{n_points});
  PointsAlpaka<2> d_points(queue, n_points);

  const float dc{20.f}, rhoc{10.f}, outlier{20.f};
  const int pPBin{128};

  // construct the tiles
  alignas(TilesAlpaka<2>) unsigned char buffer[2 * sizeof(TilesAlpaka<2>)];
  auto tiles = new (buffer) TilesAlpaka<2>(queue, n_points, pPBin);

  CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue, tiles);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
  tiles->~TilesAlpaka<2>();

  auto truth = read_output<2>("./sissa_1000_truth.csv");

  CHECK(clue::validate_results(std::span{results.data(), n_points},
                               std::span{truth.data(), n_points}));
}
