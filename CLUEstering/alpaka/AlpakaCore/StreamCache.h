#ifndef AlpakaCore_StreamCache_h
#define AlpakaCore_StreamCache_h

#include <memory>
#include <vector>

#include "alpakaConfig.h"
#include "getDeviceIndex.h"
#include "Framework/ReusableObjectHolder.h"

namespace cms::alpakatools {

  template <typename Queue>
  class StreamCache {
    using Device = alpaka::Dev<Queue>;
    using Platform = alpaka::Pltf<Device>;

  public:
    // StreamCache should be constructed by the first call to
    // getStreamCache() only if we have CUDA devices present
    StreamCache() : cache_(alpaka::getDevCount<Platform>()) {}

    // Gets a (cached) CUDA stream for the current device. The stream
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    ALPAKA_FN_HOST std::shared_ptr<Queue> get(Device const& dev) {
      return cache_[cms::alpakatools::getDeviceIndex(dev)].makeOrGet(
          [dev]() { return std::make_unique<Queue>(dev); });
    }

  private:
    // not thread safe, intended to be called only from CUDAService destructor
    void clear() {
      // Reset the contents of the caches, but leave an
      // edm::ReusableObjectHolder alive for each device. This is needed
      // mostly for the unit tests, where the function-static
      // StreamCache lives through multiple tests (and go through
      // multiple shutdowns of the framework).
      cache_.clear();
      cache_.resize(alpaka::getDevCount<Platform>());
    }

    std::vector<edm::ReusableObjectHolder<Queue>> cache_;
  };

  // Gets the global instance of a StreamCache
  // This function is thread safe
  template <typename Queue>
  StreamCache<Queue>& getStreamCache() {
    // the public interface is thread safe
    static StreamCache<Queue> cache;
    return cache;
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_StreamCache_h
