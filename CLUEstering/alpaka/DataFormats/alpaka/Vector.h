
#pragma once

//
// Author: Simone Balducci
// Note: Based on Felice's VecArray
//

namespace clue {

  // TODO: Use a caching allocator to improve dynamic allocation
  template <typename T>
  class Vector {
  private:
    T* m_data;
    uint32_t m_size;
    uint32_t m_capacity;

  public:
    Vector() = default;
    Vector(uint32_t size) : m_data(new T[size]), m_size{0}, m_capacity{size} {}
    Vector(T* data, uint32_t size) : m_data(data), m_size{0}, m_capacity{size} {}

    inline constexpr int push_back_unsafe(const T& element) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < m_capacity) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    // thread-safe version of the vector, when used in a CUDA kernel
    template <typename TAcc>
    ALPAKA_FN_ACC int push_back(const TAcc& acc, const T& element) {
      auto previousSize =
          alpaka::atomicAdd(acc, &m_size, 1u, alpaka::hierarchy::Blocks{});
      if (previousSize < m_capacity) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, 1u, alpaka::hierarchy::Blocks{});
        assert(("Too few elemets reserved"));
        return -1;
      }
    }

    inline constexpr T const* begin() const { return m_data; }
    inline constexpr T const* end() const { return m_data + m_size; }
    inline constexpr T* begin() { return m_data; }
    inline constexpr T* end() { return m_data + m_size; }

    inline constexpr T const* data() const { return m_data; }
    inline constexpr T* data() { return m_data; }

    inline constexpr const T& operator[](int i) const { return m_data[i]; }
    inline constexpr T& operator[](int i) { return m_data[i]; }

    inline constexpr size_t size() const { return m_size; }
    inline constexpr size_t capacity() const { return m_capacity; }

    inline constexpr bool empty() const { return 0 == m_size; }
    inline constexpr bool full() const { return m_capacity == m_size; }

    inline constexpr void reset() { m_size = 0; }
    inline constexpr void resize(T* data, int size) {
      m_data = data;
      m_size = 0;
      m_capacity = size;
    }
    inline constexpr void resize(int size) { m_size = size; }
    inline constexpr void reserve(uint32_t size) {
      m_capacity = size;
      m_size = 0;

      // move data to new location
      T* new_data = new T[size];
      m_data = new_data;
      new_data = nullptr;
    }
  };
};  // namespace clue
