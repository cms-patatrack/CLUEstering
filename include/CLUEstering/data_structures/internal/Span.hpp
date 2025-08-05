
#pragma once

#include <cstdint>

namespace clue {

  template <typename T>
  struct Span {
    T* buf;
    int32_t m_size;

    ALPAKA_FN_ACC T* data() { return buf; }
    ALPAKA_FN_ACC const T* data() const { return buf; }

    ALPAKA_FN_ACC auto size() const { return m_size; }

    ALPAKA_FN_ACC T& operator[](int i) { return buf[i]; }
    ALPAKA_FN_ACC const T& operator[](int i) const { return buf[i]; }
  };

};  // namespace clue
