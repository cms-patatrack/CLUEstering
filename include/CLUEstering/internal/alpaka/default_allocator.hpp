
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>
#include <concepts>

CLUE_NAMESPACE_BEGIN

/// @brief Tag type selecting the native alpaka allocations for the device buffers
struct DefaultAllocator {};

namespace concepts {

  template <typename TAllocator>
  concept allocator =
      std::same_as<TAllocator, DefaultAllocator> || alpaka::concepts::Allocator<TAllocator>;

  template <typename TAllocator>
  concept external_allocator = allocator<TAllocator> && !std::same_as<TAllocator, DefaultAllocator>;

}  // namespace concepts

namespace internal {

  // Allocates a stream-ordered device buffer, either through the native alpaka allocation or
  // through the external allocator. Both return the same buffer type.
  // TODO: remove once alpaka accepts a default allocator in allocAsyncBuf
  template <typename TElem,
            concepts::queue TQueue,
            typename TExtent,
            concepts::allocator TAllocator = DefaultAllocator>
  auto alloc_device_buffer(TQueue& queue,
                           const TExtent& extent,
                           const TAllocator& allocator = TAllocator{}) {
    if constexpr (std::same_as<TAllocator, DefaultAllocator>) {
      return alpaka::allocAsyncBuf<TElem>(queue, extent);
    } else if constexpr (std::same_as<TQueue, alpaka::QueueCpuBlocking>) {
      // alpaka 2.2 cannot enqueue the deallocation of an allocator-aware stream-ordered buffer on
      // a blocking CPU queue. On a blocking queue the enqueued work is already completed, so a
      // regular buffer is equivalent.
      // TODO: remove once fixed in alpaka
      return alpaka::allocBuf<TElem>(alpaka::getDev(queue), extent, allocator);
    } else {
      return alpaka::allocAsyncBuf<TElem>(queue, extent, allocator);
    }
  }

}  // namespace internal

CLUE_NAMESPACE_END
