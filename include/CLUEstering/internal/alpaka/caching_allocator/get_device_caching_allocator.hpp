
#pragma once

#include <optional>
#include <mutex>
#include <vector>

#include "CLUEstering/internal/alpaka/caching_allocator/allocator_config.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/caching_allocator.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/internal/alpaka/get_device_index.hpp"

namespace clue {

  namespace detail {

    template <typename TDevice, typename TQueue>
    auto allocate_device_allocators() {
      using Allocator = CachingAllocator<TDevice, TQueue>;
      auto const& devices = clue::enumerate<alpaka::Platform<TDevice>>();
      auto const size = devices.size();

      // allocate the storage for the objects
      auto ptr = std::allocator<Allocator>().allocate(size);

      // construct the objects in the storage
      for (size_t index = 0; index < size; ++index) {
        new (ptr + index) Allocator(devices[index],
                                    config::binGrowth,
                                    config::minBin,
                                    config::maxBin,
                                    config::maxCachedBytes,
                                    config::maxCachedFraction,
                                    true,    // reuseSameQueueAllocations
                                    false);  // debug
      }

      // use a custom deleter to destroy all objects and deallocate the memory
      auto deleter = [size](Allocator* pointer) {
        for (size_t i = size; i > 0; --i) {
          (pointer + i - 1)->~Allocator();
        }
        std::allocator<Allocator>().deallocate(pointer, size);
      };

      return std::unique_ptr<Allocator[], decltype(deleter)>(ptr, deleter);
    }

  }  // namespace detail

  template <typename TDevice, typename TQueue>
  inline CachingAllocator<TDevice, TQueue>& getDeviceCachingAllocator(TDevice const& device) {
    // initialise all allocators, one per device
    static auto allocators = detail::allocate_device_allocators<TDevice, TQueue>();

    size_t const index = getDeviceIndex(device);

    std::vector<TDevice> devs = alpaka::getDevs(alpaka::Platform<TDevice>{});

    assert(index < clue::enumerate<alpaka::Platform<TDevice>>().size());

    // the public interface is thread safe
    return allocators[index];
  }

}  // namespace clue
