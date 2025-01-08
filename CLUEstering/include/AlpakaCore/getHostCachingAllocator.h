
#pragma once

#include "AllocatorConfig.h"
#include "CachingAllocator.h"
#include "alpakaDevices.h"

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
