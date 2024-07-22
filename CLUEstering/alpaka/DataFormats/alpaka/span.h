
#ifndef span_h
#define span_h

#include <cstdint>
#include <type_traits>

namespace clue {

  template <typename T>
  class span {
  private:
    T* m_data;
    uint32_t m_size;

  public:
    span() = delete;
    template <typename E, typename = std::enable_if_t<std::is_convertible<E, T>::value>>
    span(E* data, uint32_t size) : m_data{data}, m_size{size} {}

    ALPAKA_FN_HOST_ACC inline T* data() { return m_data; }
    ALPAKA_FN_HOST_ACC inline const T* data() const { return m_data; }

    ALPAKA_FN_HOST_ACC inline uint32_t size() const { return m_size; }

    ALPAKA_FN_HOST_ACC inline T& operator[](uint32_t i) { return m_data[i]; }
    ALPAKA_FN_HOST_ACC inline const T& operator[](uint32_t i) const { return m_data[i]; }
  };

};  // namespace clue

#endif
