#ifndef AlpakaCore_getHostCachingAllocator_h
#define AlpakaCore_getHostCachingAllocator_h

#include "AllocatorConfig.h"
#include "CachingAllocator.h"
#include "alpakaDevices.h"

namespace cms::alpakatools {

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

}  // namespace cms::alpakatools

#endif  // AlpakaCore_getHostCachingAllocator_h
