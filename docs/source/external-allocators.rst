Using external allocators
=========================

Since release ``2.14.0`` CLUEstering supports the option to specify an external allocator for
the allocation of the internal memory buffers on the device. This can be used, for example,
to make use of a caching allocator, reusing memory previously allocated by other algorithms.

Wrapping the allocator
----------------------
Since there is not a single universally accepted API convention for device caching allocators,
*alpaka* requires the allocators to be wrapped in a struct respecting the ``Allocator`` concept:

.. code-block:: cpp

  namespace alpaka::concepts {
    template<typename TAllocator>
    concept Allocator = std::copyable<TAllocator>
                        && requires(TAllocator alloc, void* ptr, std::size_t bytes, std::size_t align) {
                               {
                                   alloc.allocate(bytes, align)
                               } -> std::same_as<void*>;
                               alloc.deallocate(ptr);
                           };
  }

Taking the ``cub::CachingAllocator`` as an example, the wrapper can be implemented in the following way:

.. code-block:: cpp

  // Minimal wrapper adapting cub::CachingDeviceAllocator to the allocator interface
  struct AllocatorWrapper {
    cub::CachingDeviceAllocator* allocator;

    void* allocate(std::size_t bytes, [[maybe_unused]] std::size_t align) {
      void* ptr = nullptr;
      if (allocator->DeviceAllocate(&ptr, bytes) != cudaSuccess)
        throw std::runtime_error("DeviceAllocate failed");
      return ptr;
    }

    void deallocate(void* ptr) {
      if (allocator->DeviceFree(ptr) != cudaSuccess)
        throw std::runtime_error("DeviceFree failed");
    }
  };

The allocator wrapper can then be passed to the constructor of the ``Clusterer``, where it will be saved
and automatically used internally for the allocations on the device:

.. code-block:: cpp

   auto allocator_wrapper = AllocatorWrapper(&allocator);
   auto clusterer = clue::Clusterer<2, float, AllocatorWrapper>(allocator_wrapper, density_radius, min_density);
   clusterer.make_clusters(queue, host_points);   // here the allocator is used for allocating the device points,
                                                  // as well as all the internal buffers and data structures
