
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>

namespace clue::internal {

  class SeedArrayView {
    int32_t* m_data;
    std::size_t* m_size;
    std::size_t m_capacity;

  public:
    ALPAKA_FN_HOST_ACC constexpr SeedArrayView(int32_t* data,
                                               std::size_t* size,
                                               std::size_t capacity)
        : m_data{data}, m_size{size}, m_capacity{capacity} {}

    ALPAKA_FN_ACC constexpr auto& operator[](std::size_t index) { return m_data[index]; }
    ALPAKA_FN_ACC constexpr const auto& operator[](std::size_t index) const {
      return m_data[index];
    }

    ALPAKA_FN_ACC constexpr auto size() const {
      // NOTE: not thread-safe
      // Could restrict this to KernelAssignClusters, but maybe that's an overkill
      return *m_size;
    }

    template <clue::concepts::accelerator TAcc>
    ALPAKA_FN_ACC constexpr void push_back(const TAcc& acc, int32_t value) {
      auto prev = alpaka::atomicAdd(acc, m_size, 1ul);
      if (prev < m_capacity) {
        m_data[prev] = value;
      } else {
        alpaka::atomicSub(acc, m_size, 1ul);
      }
    }
  };

  template <clue::concepts::device TDev = clue::Device>
  class SeedArray {
  private:
    clue::device_buffer<TDev, int32_t[]> m_buffer;
    clue::device_buffer<TDev, std::size_t> m_dsize;
    std::optional<std::size_t> m_size;
    std::size_t m_capacity;
    SeedArrayView m_view;

  public:
    template <clue::concepts::queue TQueue>
    SeedArray(TQueue& queue, std::size_t size)
        : m_buffer{clue::make_device_buffer<int32_t[]>(queue, size)},
          m_dsize{clue::make_device_buffer<std::size_t>(queue)},
          m_size{std::nullopt},
          m_capacity{size},
          m_view{m_buffer.data(), m_dsize.data(), m_capacity} {}

    ALPAKA_FN_HOST constexpr auto capacity() const { return m_capacity; }

    template <clue::concepts::queue TQueue>
    ALPAKA_FN_HOST auto size(TQueue& queue) {
      if (!m_size.has_value()) {
        m_size = std::make_optional<std::size_t>();
        alpaka::memcpy(queue, clue::make_host_view(*m_size), m_dsize);
        alpaka::wait(queue);
      }
      return *m_size;
    }

    template <clue::concepts::queue TQueue>
    ALPAKA_FN_HOST auto reset(TQueue& queue) {
      m_size = std::nullopt;
      alpaka::memset(queue, m_dsize, 0u);
    }

    ALPAKA_FN_HOST const auto& view() const { return m_view; }
    ALPAKA_FN_HOST auto& view() { return m_view; }
  };

}  // namespace clue::internal
