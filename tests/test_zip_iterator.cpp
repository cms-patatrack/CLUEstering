
#include "CLUEstering/internal/nostd/zip_iterator.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("Test zip iterator with two containers of the same type") {
  const std::size_t size = 100;
  std::vector<int> v1(size);
  std::vector<int> v2(size);
  std::iota(v1.begin(), v1.end(), 0);
  std::iota(v2.begin(), v2.end(), size);

  auto zip_begin = clue::nostd::zip(v1.begin(), v2.begin());
  auto zip_end = zip_begin + size;

  SUBCASE("Test iterator dereference and accessor") {
    auto element = *zip_begin;
    CHECK(std::is_same_v<decltype(element), std::tuple<int&, int&>>);
    CHECK(std::get<0>(element) == 0);
    CHECK(std::get<1>(element) == size);
    auto second_element = zip_begin[1];
    CHECK(std::is_same_v<decltype(second_element), std::tuple<int&, int&>>);
    CHECK(std::get<0>(second_element) == 1);
    CHECK(std::get<1>(second_element) == size + 1);

    const auto const_zip_begin = clue::nostd::zip(v1.cbegin(), v2.cbegin());
    CHECK(std::is_same_v<decltype(*const_zip_begin), std::tuple<int, int>>);
    auto const_element = *const_zip_begin;
    CHECK(std::get<0>(const_element) == 0);
    CHECK(std::get<1>(const_element) == size);
    auto const_second_element = const_zip_begin[1];
    CHECK(std::is_same_v<decltype(const_second_element), std::tuple<int, int>>);
    CHECK(std::get<0>(const_second_element) == 1);
    CHECK(std::get<1>(const_second_element) == size + 1);
  }
  SUBCASE("Test iterator increment and decrement") {
    auto it = zip_begin;
    ++it;
    CHECK(std::get<0>(*it) == 1);
    CHECK(std::get<1>(*it) == size + 1);
    --it;
    CHECK(std::get<0>(*it) == 0);
    CHECK(std::get<1>(*it) == size);
    CHECK(it == zip_begin);
  }
  SUBCASE("Test iterator advance and addition/subtraction") {
    auto it = zip_begin;
    it += 10;
    CHECK(std::get<0>(*it) == 10);
    CHECK(std::get<1>(*it) == size + 10);
    it = it + 20;
    CHECK(std::get<0>(*it) == 30);
    CHECK(std::get<1>(*it) == size + 30);
    it = it - 5;
    CHECK(std::get<0>(*it) == 25);
    CHECK(std::get<1>(*it) == size + 25);
    it -= 25;
    CHECK(it == zip_begin);
  }

  std::for_each(zip_begin, zip_end, [size](auto&& tuple) {
    auto a = std::get<0>(tuple);
    auto b = std::get<1>(tuple);
    CHECK(b - a == size);
  });
  std::for_each(zip_begin, zip_end, [size, i = 0](auto&& tuple) mutable {
    auto a = std::get<0>(tuple);
    auto b = std::get<1>(tuple);
    CHECK(a == i);
    CHECK(b == i + size);
    ++i;
  });
}
