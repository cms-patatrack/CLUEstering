
#pragma once

namespace clue {

  template <typename T>
  struct Span {
    T* buf;
    int m_size;

    ALPAKA_FN_ACC T* data() { return buf; }
    ALPAKA_FN_ACC const T* data() const { return buf; }
    ALPAKA_FN_ACC int size() const { return m_size; }

	ALPAKA_FN_ACC T& operator[](int i) { return buf[i]; }
	ALPAKA_FN_ACC const T& operator[](int i) const { return buf[i]; }
  };

};  // namespace clue
