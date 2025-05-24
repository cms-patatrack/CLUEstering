
#include "CLUEstering/DataFormats/PointsHost.hpp"
#include "CLUEstering/DataFormats/PointsDevice.hpp"

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "doctest.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

TEST_CASE("Test host points with internal allocation") {
  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  const uint32_t size = 1000;
  clue::PointsDevice<2, Device> h_points(queue, size);
  auto view = h_points.view();

  CHECK(view->n == size);
  SUBCASE("Set from view") {
    auto coords = view->coords;
    auto weights = view->weight;
    auto cluster_indexes = view->cluster_index;
    auto is_seed = view->is_seed;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);
    std::fill(is_seed, is_seed + size, 3);

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, 2 * size),
        std::views::iota(2000) | std::views::take(2 * size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
    std::ranges::for_each(std::span(is_seed, size), [](auto x) { CHECK(x == 3); });
  }
}

TEST_CASE("Test host points with external allocation of whole buffer") {
  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  const uint32_t size = 1000;
  const auto bytes = clue::soa::host::computeSoASize<2>(size);
  auto buffer = clue::make_device_buffer<std::byte[]>(queue, bytes);

  clue::PointsHost<2> h_points(queue, size, std::span(buffer.data(), bytes));
  auto view = h_points.view();

  CHECK(view->n == size);
  SUBCASE("Set from view") {
    auto coords = view->coords;
    auto weights = view->weight;
    auto cluster_indexes = view->cluster_index;
    auto is_seed = view->is_seed;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);
    std::fill(is_seed, is_seed + size, 3);

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, 2 * size),
        std::views::iota(2000) | std::views::take(2 * size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
    std::ranges::for_each(std::span(is_seed, size), [](auto x) { CHECK(x == 3); });
  }
}

TEST_CASE("Test host points with external allocation passing the two buffers as spans") {
  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  const uint32_t size = 1000;
  auto input = clue::make_device_buffer<float[]>(queue, 3 * size);
  auto output = clue::make_device_buffer<int[]>(queue, 2 * size);

  clue::PointsHost<2> h_points(
      queue, size, std::span(input.data(), 3 * size), std::span(output.data(), 2 * size));
  auto view = h_points.view();

  CHECK(view->n == size);
  SUBCASE("Set from view") {
    auto coords = view->coords;
    auto weights = view->weight;
    auto cluster_indexes = view->cluster_index;
    auto is_seed = view->is_seed;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);
    std::fill(is_seed, is_seed + size, 3);

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, 2 * size),
        std::views::iota(2000) | std::views::take(2 * size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
    std::ranges::for_each(std::span(is_seed, size), [](auto x) { CHECK(x == 3); });
  }
}

TEST_CASE("Test host points with external allocation passing the two buffers as pointers") {
  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  const uint32_t size = 1000;
  auto input = clue::make_device_buffer<float[]>(queue, 3 * size);
  auto output = clue::make_device_buffer<int[]>(queue, 2 * size);

  clue::PointsHost<2> h_points(queue, size, input.data(), output.data());
  auto view = h_points.view();

  CHECK(view->n == size);
  SUBCASE("Set from view") {
    auto coords = view->coords;
    auto weights = view->weight;
    auto cluster_indexes = view->cluster_index;
    auto is_seed = view->is_seed;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);
    std::fill(is_seed, is_seed + size, 3);

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, 2 * size),
        std::views::iota(2000) | std::views::take(2 * size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
    std::ranges::for_each(std::span(is_seed, size), [](auto x) { CHECK(x == 3); });
  }
}

TEST_CASE("Test host points with external allocation passing four buffers as spans") {
  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  const uint32_t size = 1000;
  auto coords = clue::make_device_buffer<float[]>(queue, 2 * size);
  auto weights = clue::make_device_buffer<int[]>(queue, size);
  auto cluster_ids = clue::make_device_buffer<int[]>(queue, size);
  auto b_isseed = clue::make_device_buffer<int[]>(queue, size);

  clue::PointsHost<2> h_points(queue,
                               size,
                               std::span(coords.data(), 2 * size),
                               std::span(weights.data(), size),
                               std::span(cluster_ids.data(), size),
                               std::span(b_isseed.data(), size));
  auto view = h_points.view();

  CHECK(view->n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords();
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();
    auto is_seed = h_points.isSeed();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);
    std::fill(is_seed.begin(), is_seed.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, size * 2)));
    CHECK(std::ranges::equal(weights, std::span<float>(view->weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view->cluster_index, size)));
    CHECK(std::ranges::equal(is_seed, std::span<int>(view->is_seed, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0, (int)(2 * size)) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0, (int)size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
    std::ranges::for_each(is_seed, [](auto x) { CHECK(x == 0); });
  }

  SUBCASE("Set from view") {
    auto coords = view->coords;
    auto weights = view->weight;
    auto cluster_indexes = view->cluster_index;
    auto is_seed = view->is_seed;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);
    std::fill(is_seed, is_seed + size, 3);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, 2 * size), h_points.coords()));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));
    CHECK(std::ranges::equal(std::span(is_seed, size), h_points.isSeed()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, 2 * size),
        std::views::iota(2000) | std::views::take(2 * size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
    std::ranges::for_each(std::span(is_seed, size), [](auto x) { CHECK(x == 3); });
  }
}

TEST_CASE("Test host points with external allocation passing four buffers as pointers") {
  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  const uint32_t size = 1000;
  auto coords = clue::make_device_buffer<float[]>(queue, 2 * size);
  auto weights = clue::make_device_buffer<int[]>(queue, size);
  auto cluster_ids = clue::make_device_buffer<int[]>(queue, size);
  auto b_isseed = clue::make_device_buffer<int[]>(queue, size);

  clue::PointsHost<2> h_points(
      queue, size, coords.data(), weights.data(), cluster_ids.data(), b_isseed.data());
  auto view = h_points.view();

  CHECK(view->n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords();
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();
    auto is_seed = h_points.isSeed();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);
    std::fill(is_seed.begin(), is_seed.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, size * 2)));
    CHECK(std::ranges::equal(weights, std::span<float>(view->weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view->cluster_index, size)));
    CHECK(std::ranges::equal(is_seed, std::span<int>(view->is_seed, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0, (int)(2 * size)) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0, (int)size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
    std::ranges::for_each(is_seed, [](auto x) { CHECK(x == 0); });
  }

  SUBCASE("Set from view") {
    auto coords = view->coords;
    auto weights = view->weight;
    auto cluster_indexes = view->cluster_index;
    auto is_seed = view->is_seed;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);
    std::fill(is_seed, is_seed + size, 3);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, 2 * size), h_points.coords()));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));
    CHECK(std::ranges::equal(std::span(is_seed, size), h_points.isSeed()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, 2 * size),
        std::views::iota(2000) | std::views::take(2 * size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
    std::ranges::for_each(std::span(is_seed, size), [](auto x) { CHECK(x == 3); });
  }
}
