
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/utils/get_device.hpp"

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "doctest.h"

TEST_CASE("Test host points with internal allocation") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  clue::PointsHost<2> h_points(queue, size);
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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

TEST_CASE("Test host points with external allocation of whole buffer") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<std::byte> buffer(clue::soa::host::computeSoASize<2>(size));

  clue::PointsHost<2> h_points(queue, size, std::span(buffer.data(), buffer.size()));
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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

TEST_CASE("Test host points with external allocation passing two buffers as spans") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> input(3 * size);
  std::vector<int> output(2 * size);

  clue::PointsHost<2> h_points(
      queue, size, std::span(input.data(), input.size()), std::span(output.data(), output.size()));
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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

TEST_CASE("Test host points with external allocation passing two buffers as vectors") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> input(3 * size);
  std::vector<int> output(2 * size);

  clue::PointsHost<2> h_points(queue, size, input, output);
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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

TEST_CASE("Test host points with external allocation passing two buffers as pointers") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> input(3 * size);
  std::vector<int> output(2 * size);

  clue::PointsHost<2> h_points(queue, size, input.data(), output.data());
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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

TEST_CASE("Test host points with external allocation passing four buffers as spans") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> coords(2 * size);
  std::vector<float> weights(size);
  std::vector<int> cluster_ids(size);
  std::vector<int> v_isseed(size);

  clue::PointsHost<2> h_points(queue,
                               size,
                               std::span(coords.data(), coords.size()),
                               std::span(weights.data(), weights.size()),
                               std::span(cluster_ids.data(), cluster_ids.size()),
                               std::span(v_isseed.data(), v_isseed.size()));
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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

TEST_CASE("Test host points with external allocation passing four buffers as vectors") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> coords(2 * size);
  std::vector<float> weights(size);
  std::vector<int> cluster_ids(size);
  std::vector<int> v_isseed(size);

  clue::PointsHost<2> h_points(queue, size, coords, weights, cluster_ids, v_isseed);
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> coords(2 * size);
  std::vector<float> weights(size);
  std::vector<int> cluster_ids(size);
  std::vector<int> v_isseed(size);

  clue::PointsHost<2> h_points(
      queue, size, coords.data(), weights.data(), cluster_ids.data(), v_isseed.data());
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
    CHECK(std::ranges::equal(coords, std::span<float>(view->coords, 2 * size)));
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
