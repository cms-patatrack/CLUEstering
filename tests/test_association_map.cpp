
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test binary association map") {
  const auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
  clue::Queue queue(host);

  const int32_t size = 1000;
  auto associations = clue::make_host_buffer<int32_t[]>(queue, size);
  std::ranges::transform(
      std::views::iota(0, size), associations.data(), [](auto x) -> int32_t { return x % 2 == 0; });
  auto map =
      clue::internal::make_associator(queue, std::span<int32_t>(associations.data(), size), size);

  SUBCASE("Check size") { CHECK(map.size() == 2); }
  SUBCASE("Check extents") {
    CHECK(map.extents().values == size);
    CHECK(map.extents().keys == 2);
  }
  SUBCASE("Test contains") {
    CHECK(map.contains(0));
    CHECK(map.contains(1));
  }
  SUBCASE("Test count") {
    CHECK(map.count(0) == size / 2);
    CHECK(map.count(1) == size / 2);
  }

  SUBCASE("Test iterators") {
    auto begin = map.begin();
    auto end = map.end();
    CHECK(std::distance(begin, end) == size);
    CHECK(std::distance(begin, begin + size / 2) == size / 2);
    CHECK(std::distance(end - size / 2, end) == size / 2);
  }
  SUBCASE("Test lower_bound and upper_bound") {
    CHECK(std::distance(map.lower_bound(0), map.upper_bound(0)) == size / 2);
    CHECK(std::distance(map.lower_bound(1), map.upper_bound(1)) == size / 2);
  }
  SUBCASE("Test equal_range") {
    CHECK(std::distance(map.equal_range(0).first, map.equal_range(0).second) == size / 2);
    CHECK(std::distance(map.equal_range(1).first, map.equal_range(1).second) == size / 2);
  }
  SUBCASE("Test accessor to underlying containers") {
    auto containers = map.extract();
    const auto& keys = containers.keys;
    const auto& values = containers.values;
    CHECK(keys[0] == 0);
    CHECK(keys[1] == size / 2);
    CHECK(keys[2] == size);
    CHECK(values[0] % 2 != 0);
    CHECK(values[size - 1] % 2 == 0);
  }
}

TEST_CASE("Test throwing conditions") {
  const auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
  clue::Queue queue(host);

  const int32_t size = 1000;
  auto associations = clue::make_host_buffer<int32_t[]>(queue, size);
  std::ranges::transform(
      std::views::iota(0, size), associations.data(), [](auto x) -> int32_t { return x % 2 == 0; });

  SUBCASE("Test construction throwing conditions") {
    CHECK_THROWS(
        clue::internal::make_associator(queue, std::span<int32_t>(associations.data(), 0), 0));
  }

  auto map =
      clue::internal::make_associator(queue, std::span<int32_t>(associations.data(), size), size);
  SUBCASE("Test count throwing conditions") {
    CHECK_THROWS(map.count(-1));
    CHECK_THROWS(map.count(2));
    CHECK_THROWS(map.count(3));
  }
  SUBCASE("Test contains throwing conditions") {
    CHECK_THROWS(map.contains(-1));
    CHECK_THROWS(map.contains(2));
    CHECK_THROWS(map.contains(3));
  }
  SUBCASE("Test lower_bound throwing conditions") {
    CHECK_THROWS(map.lower_bound(-1));
    CHECK_THROWS(map.lower_bound(2));
    CHECK_THROWS(map.lower_bound(3));
  }
  SUBCASE("Test upper_bound throwing conditions") {
    CHECK_THROWS(map.upper_bound(-1));
    CHECK_THROWS(map.upper_bound(2));
    CHECK_THROWS(map.upper_bound(3));
  }
  SUBCASE("Test equal_range throwing conditions") {
    CHECK_THROWS(map.equal_range(-1));
    CHECK_THROWS(map.equal_range(2));
    CHECK_THROWS(map.equal_range(3));
  }

  const auto const_map =
      clue::internal::make_associator(queue, std::span<int32_t>(associations.data(), size), size);
  SUBCASE("Test lower_bound throwing conditions") {
    CHECK_THROWS(map.lower_bound(-1));
    CHECK_THROWS(map.lower_bound(2));
    CHECK_THROWS(map.lower_bound(3));
  }
  SUBCASE("Test upper_bound throwing conditions") {
    CHECK_THROWS(map.upper_bound(-1));
    CHECK_THROWS(map.upper_bound(2));
    CHECK_THROWS(map.upper_bound(3));
  }
  SUBCASE("Test equal_range throwing conditions") {
    CHECK_THROWS(map.equal_range(-1));
    CHECK_THROWS(map.equal_range(2));
    CHECK_THROWS(map.equal_range(3));
  }
}

TEST_CASE("Test binary host_associator") {
  const int32_t size = 1000;

  auto associations = clue::make_host_buffer<int32_t[]>(size);
  std::ranges::transform(
      std::views::iota(0, size), associations.data(), [](auto x) -> int32_t { return x % 2 == 0; });
  auto map = clue::internal::make_associator(std::span<int32_t>(associations.data(), size), size);

  SUBCASE("Check size") { CHECK(map.size() == 2); }
  SUBCASE("Check extents") {
    CHECK(map.extents().values == size);
    CHECK(map.extents().keys == 2);
  }
  SUBCASE("Test contains") {
    CHECK(map.contains(0));
    CHECK(map.contains(1));
  }
  SUBCASE("Test count") {
    CHECK(map.count(0) == size / 2);
    CHECK(map.count(1) == size / 2);
  }

  SUBCASE("Test iterators") {
    auto begin = map.begin();
    auto end = map.end();
    CHECK(std::distance(begin, end) == size);
    CHECK(std::distance(begin, begin + size / 2) == size / 2);
    CHECK(std::distance(end - size / 2, end) == size / 2);
  }
  SUBCASE("Test lower_bound and upper_bound") {
    CHECK(std::distance(map.lower_bound(0), map.upper_bound(0)) == size / 2);
    CHECK(std::distance(map.lower_bound(1), map.upper_bound(1)) == size / 2);
  }
  SUBCASE("Test equal_range") {
    CHECK(std::distance(map.equal_range(0).first, map.equal_range(0).second) == size / 2);
    CHECK(std::distance(map.equal_range(1).first, map.equal_range(1).second) == size / 2);
  }
}

#else

int main() {}

#endif
