
#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/utils/get_device.hpp"
#include "CLUEstering/utils/get_queue.hpp"

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

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
  }
}

TEST_CASE("Test host points with external allocation of whole buffer") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<std::byte> buffer(clue::soa::host::computeSoASize<2>(size));

  clue::PointsHost<2> h_points(queue, size, std::span(buffer.data(), buffer.size()));
  auto view = h_points.view();

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
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

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
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

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
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

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
  }
}

TEST_CASE("Test host points with external allocation passing four buffers as spans") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> coords(2 * size);
  std::vector<float> weights(size);
  std::vector<int> cluster_ids(size);

  clue::PointsHost<2> h_points(queue,
                               size,
                               std::span(coords.data(), coords.size()),
                               std::span(weights.data(), weights.size()),
                               std::span(cluster_ids.data(), cluster_ids.size()));
  auto view = h_points.view();

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
  }
}

TEST_CASE("Test host points with external allocation passing four buffers as vectors") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> coords(2 * size);
  std::vector<float> weights(size);
  std::vector<int> cluster_ids(size);

  clue::PointsHost<2> h_points(queue, size, coords, weights, cluster_ids);
  auto view = h_points.view();

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
  }
}

TEST_CASE("Test host points with external allocation passing four buffers as pointers") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const uint32_t size = 1000;
  std::vector<float> coords(2 * size);
  std::vector<float> weights(size);
  std::vector<int> cluster_ids(size);

  clue::PointsHost<2> h_points(queue, size, coords.data(), weights.data(), cluster_ids.data());
  auto view = h_points.view();

  CHECK(view.n == size);
  SUBCASE("Set from host span") {
    auto coords = h_points.coords(0);
    auto weights = h_points.weights();
    auto cluster_indexes = h_points.clusterIndexes();

    std::iota(coords.begin(), coords.end(), 0.f);
    std::fill(weights.begin(), weights.end(), 1.f);
    std::iota(cluster_indexes.begin(), cluster_indexes.end(), 0);

    // compare with content of the view
    CHECK(std::ranges::equal(coords, std::span<float>(view.coords[0], size)));
    CHECK(std::ranges::equal(weights, std::span<float>(view.weight, size)));
    CHECK(std::ranges::equal(cluster_indexes, std::span<int>(view.cluster_index, size)));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        coords, std::views::iota(0) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(cluster_indexes, std::views::iota(0) | std::views::take(size)));
    std::ranges::for_each(weights, [](auto x) { CHECK(x == 1.f); });
  }

  SUBCASE("Set from view") {
    auto coords = view.coords[0];
    auto weights = view.weight;
    auto cluster_indexes = view.cluster_index;

    std::iota(coords, coords + 2 * size, 2000.f);
    std::fill(weights, weights + size, 2.f);
    std::iota(cluster_indexes, cluster_indexes + size, 2000);

    // compare with content of the view
    CHECK(std::ranges::equal(std::span(coords, size), h_points.coords(0)));
    CHECK(std::ranges::equal(std::span(weights, size), h_points.weights()));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size), h_points.clusterIndexes()));

    // check content
    auto to_float = [](int i) -> float { return static_cast<float>(i); };
    CHECK(std::ranges::equal(
        std::span(coords, size),
        std::views::iota(2000) | std::views::take(size) | std::views::transform(to_float)));
    CHECK(std::ranges::equal(std::span(cluster_indexes, size),
                             std::views::iota(2000) | std::views::take(size)));
    std::ranges::for_each(std::span(weights, size), [](auto x) { CHECK(x == 2.f); });
  }
}

TEST_CASE("Test point accessor") {
  auto queue = clue::get_queue(0u);
  const uint32_t size = 1000;

  clue::PointsHost<2> points(queue, size);
  std::iota(points.coords(0).begin(), points.coords(1).end(), 0.f);
  std::fill(points.weights().begin(), points.weights().end(), 1.f);
  std::iota(points.clusterIndexes().begin(), points.clusterIndexes().end(), 0);

  SUBCASE("Test point methods") {
    auto point = points[0];
    CHECK(point[0] == 0.f);
    CHECK(point[1] == 1000.f);
    CHECK(point.weight() == 1.f);
    CHECK(point.cluster_index() == 0);

    auto point2 = points[999];
    CHECK(point2[0] == 999.f);
    CHECK(point2[1] == 1999.f);
    CHECK(point2.weight() == 1.f);
    CHECK(point2.cluster_index() == 999);
  }
}

TEST_CASE("Test constructor throwing conditions") {
  auto queue = clue::get_queue(0u);
  CHECK_THROWS(clue::PointsHost<2>(queue, 0));
  CHECK_THROWS(clue::PointsHost<2>(queue, -5));
}

TEST_CASE("Test coordinate getter throwing conditions") {
  SUBCASE("Const points") {
    const uint32_t size = 1000;
    auto queue = clue::get_queue(0u);
    clue::PointsHost<2> points(queue, size);
    CHECK_THROWS(points.coords(3));
    CHECK_THROWS(points.coords(10));
  }
  SUBCASE("Non-const points") {
    const uint32_t size = 1000;
    auto queue = clue::get_queue(0u);
    const clue::PointsHost<2> points(queue, size);
    CHECK_THROWS(points.coords(3));
    CHECK_THROWS(points.coords(10));
  }
}
