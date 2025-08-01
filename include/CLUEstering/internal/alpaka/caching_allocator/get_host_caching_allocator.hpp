
#pragma once

#include "CLUEstering/internal/alpaka/caching_allocator/allocator_config.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/caching_allocator.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"

namespace clue {

  template <typename TQueue>
  inline CachingAllocator<alpaka_common::DevHost, TQueue>& getHostCachingAllocator() {
    // thread safe initialisation of the host allocator
    static CachingAllocator<alpaka_common::DevHost, TQueue> allocator(
        host,
        config::binGrowth,
        config::minBin,
        config::maxBin,
        config::maxCachedBytes,
        config::maxCachedFraction,
        false,   // reuseSameQueueAllocations
        false);  // debug

    // the public interface is thread safe
    return allocator;
  }

}  // namespace clue
