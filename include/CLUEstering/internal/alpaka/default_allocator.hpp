
#pragma once

#include <concepts>
#include <cstddef>

#include <alpaka/alpaka.hpp>

namespace clue {

  struct DefaultAllocator {
    void* allocate(std::size_t, std::size_t /* alignment */);
    void deallocate(void*);
  };

  namespace concepts {

    template <typename TAllocator>
    concept allocator =
        std::same_as<TAllocator, DefaultAllocator> || alpaka::concepts::Allocator<TAllocator>;

    template <typename TAllocator>
    concept external_allocator =
        allocator<TAllocator> && !std::same_as<TAllocator, DefaultAllocator>;

  }  // namespace concepts

}  // namespace clue
