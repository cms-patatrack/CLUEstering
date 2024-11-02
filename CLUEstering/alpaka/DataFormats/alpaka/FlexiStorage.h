
#pragma once

template <typename I, int S>
class FlexiStorage {
public:
  ALPAKA_FN_HOST_ACC constexpr int capacity() const { return S; }

  ALPAKA_FN_HOST_ACC constexpr I& operator[](int i) { return m_v[i]; }
  ALPAKA_FN_HOST_ACC constexpr const I& operator[](int i) const { return m_v[i]; }

  ALPAKA_FN_HOST_ACC constexpr I* data() { return m_v; }
  ALPAKA_FN_HOST_ACC constexpr I const* data() const { return m_v; }

private:
  I m_v[S];
};

template <typename I>
class FlexiStorage<I, -1> {
public:
  constexpr void init(I* v, int s) {
    m_v = v;
    m_capacity = s;
  }

  ALPAKA_FN_HOST_ACC constexpr int capacity() const { return m_capacity; }

  ALPAKA_FN_HOST_ACC constexpr I& operator[](int i) { return m_v[i]; }
  ALPAKA_FN_HOST_ACC constexpr const I& operator[](int i) const { return m_v[i]; }

  ALPAKA_FN_HOST_ACC constexpr I* data() { return m_v; }
  ALPAKA_FN_HOST_ACC constexpr I const* data() const { return m_v; }

private:
  I* m_v;
  int m_capacity;
};
