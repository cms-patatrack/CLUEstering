
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "FlexiStorage.h"
#include "AlpakaCore/alpakaWorkDiv.h"
#include "DataFormats/alpaka/span.h"

template <typename I,    // type stored in the container
          int32_t ONES,  // number of "Ones" +1. If -1 is initialized at runtime
          int32_t SIZE>  // max number of element. If -1 is initialized at runtime
class OneToManyAssociator {
public:
  using Counter = uint32_t;
  using point_index = I;

  struct View {
    OneToManyAssociator* assoc = nullptr;
    Counter* offStorage = nullptr;
    point_index* contentStorage = nullptr;
    int32_t offSize = -1;
    int32_t contentSize = -1;
  };

  ALPAKA_FN_HOST_ACC static constexpr int32_t Ones() { return ONES; }
  ALPAKA_FN_HOST_ACC static constexpr int32_t Size() { return SIZE; }
  ALPAKA_FN_HOST_ACC constexpr auto offsetCapacity() const { return off.capacity(); }
  ALPAKA_FN_HOST_ACC constexpr auto contentcapacity() const { return content.capacity(); }

  ALPAKA_FN_HOST_ACC void initStorage(View view) {
    if constexpr (Size() < 0) {
      content.init(view.contentStorage, view.contentSize);
    }
    if constexpr (Ones() < 0) {
      off.init(view.offStorage, view.offSize);
    }
  }

  ALPAKA_FN_HOST_ACC constexpr auto size() const {
    return uint32_t(off[off.capacity - 1]);
  }
  ALPAKA_FN_HOST_ACC constexpr auto size(uint32_t tile) const {
    return off[tile + 1] - off[tile];
  }

  ALPAKA_FN_HOST_ACC constexpr const point_index* begin() const { return content.data(); }
  ALPAKA_FN_HOST_ACC constexpr point_index* begin() { return content.data(); }
  ALPAKA_FN_HOST_ACC constexpr const point_index* end() const { return begin() + size(); }
  ALPAKA_FN_HOST_ACC constexpr point_index* end() { return begin() + size(); }

  ALPAKA_FN_HOST_ACC constexpr const point_index* begin(uint32_t b) const {
    return content.data() + off[b];
  }
  ALPAKA_FN_HOST_ACC constexpr point_index* begin(uint32_t b) {
    return content.data() + off[b];
  }

  ALPAKA_FN_HOST_ACC constexpr const point_index* end(uint32_t b) const {
    return content.data() + off[b + 1];
  }
  ALPAKA_FN_HOST_ACC constexpr point_index* end(uint32_t b) {
    return content.data() + off[b + 1];
  }

  ALPAKA_FN_HOST_ACC constexpr clue::span<const point_index> view(uint32_t b) const {
    return {begin(b), size(b)};
  }
  ALPAKA_FN_HOST_ACC constexpr clue::span<point_index> view(uint32_t b) {
    return {begin(b), size(b)};
  }

  FlexiStorage<Counter, ONES> off;
  FlexiStorage<point_index, SIZE> content;
};