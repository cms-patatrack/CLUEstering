
#pragma once

namespace clue {

  struct PointsView {
    float* coords;
    float* weight;
    int* cluster_index;
    int* is_seed;
    // uint8_t* wrapping;
    float* rho;
    float* delta;
    int* nearest_higher;
    uint32_t n;
  };

  namespace detail {

    template <typename T>
    concept ContiguousRange = requires(T&& t) {
      t.size();
      t.data();
    } && std::ranges::contiguous_range<T>;

    template <typename T>
    concept ArrayOrPtr = std::is_array_v<T> || std::is_pointer_v<T>;

  }  // namespace detail

  // TODO: implement for better cache use
  template <uint8_t Ndim>
  uint32_t computeAlignSoASize(uint32_t n_points);

}  // namespace clue
