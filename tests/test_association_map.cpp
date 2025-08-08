
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "association_map/build_map.hpp"

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "doctest.h"

TEST_CASE("Test binary association map") {
  const auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
  clue::Queue queue(host);

  const int32_t size = 1000;
  auto associations = clue::make_host_buffer<int32_t[]>(queue, size);
  std::ranges::transform(
      std::views::iota(0, size), associations.data(), [](auto x) -> int32_t { return x % 2 == 0; });
  auto map = clue::test::build_map(queue, std::span<int32_t>(associations.data(), size), size);

  SUBCASE("Check size") { CHECK(map.size() == 2); }
  SUBCASE("Check extents") {
    CHECK(map.extents().values == size);
    CHECK(map.extents().keys == 3);
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

#endif
