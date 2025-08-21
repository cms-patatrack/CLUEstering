
#pragma once

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/alpaka/caching_allocator/allocator_policy.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/cached_buf_alloc.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/detail/concepts.hpp"

namespace clue {

  namespace internal {
    namespace concepts {

      // bounded 1D array
      template <typename T>
      concept bounded_array =
          std::is_bounded_array_v<T> && not std::is_array_v<std::remove_extent_t<T>>;

      // unbounded 1D array
      template <typename T>
      concept unbounded_array =
          std::is_unbounded_array_v<T> && not std::is_array_v<std::remove_extent_t<T>>;

      template <typename T>
      concept scalar = not std::is_array_v<T>;

    }  // namespace concepts
  }  // namespace internal

  // for Extent, Dim1D, Idx
  using namespace alpaka_common;

  // type deduction helpers
  namespace detail {

    template <typename TDev, typename T>
    struct buffer_type {
      using type = alpaka::Buf<TDev, T, Dim0D, Idx>;
    };

    template <typename TDev, typename T>
    struct buffer_type<TDev, T[]> {
      using type = alpaka::Buf<TDev, T, Dim1D, Idx>;
    };

    template <typename TDev, typename T, int N>
    struct buffer_type<TDev, T[N]> {
      using type = alpaka::Buf<TDev, T, Dim1D, Idx>;
    };

    template <typename TDev, typename T>
    struct view_type {
      using type = alpaka::ViewPlainPtr<TDev, T, Dim0D, Idx>;
    };

    template <typename TDev, typename T>
    struct view_type<TDev, T[]> {
      using type = alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>;
    };

    template <typename TDev, typename T, int N>
    struct view_type<TDev, T[N]> {
      using type = alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>;
    };

  }  // namespace detail

  // scalar and 1-dimensional host buffers

  template <typename T>
  using host_buffer = typename detail::buffer_type<DevHost, T>::type;

  // non-cached, non-pinned, scalar and 1-dimensional host buffers

  template <internal::concepts::scalar T>
  host_buffer<T> make_host_buffer() {
    return alpaka::allocBuf<T, Idx>(host, Scalar{});
  }

  template <internal::concepts::unbounded_array T>
  host_buffer<T> make_host_buffer(Extent extent) {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(host, Vec1D{extent});
  }

  template <internal::concepts::bounded_array T>
  host_buffer<T> make_host_buffer() {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(host, Vec1D{std::extent_v<T>});
  }

  // potentially cached, pinned, scalar and 1-dimensional host buffers, associated to a work queue
  // the memory is pinned according to the device associated to the queue

  template <internal::concepts::scalar T, concepts::queue TQueue>
  host_buffer<T> make_host_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<T, Idx>(host, queue, Scalar{});
    } else {
      using Platform = alpaka::Platform<alpaka::Dev<TQueue>>;
      return alpaka::allocMappedBuf<T, Idx>(host, platform<Platform>(), Scalar{});
    }
  }

  template <internal::concepts::unbounded_array T, concepts::queue TQueue>
  host_buffer<T> make_host_buffer(TQueue const& queue, Extent extent) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(host, queue, Vec1D{extent});
    } else {
      using Platform = alpaka::Platform<alpaka::Dev<TQueue>>;
      return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(
          host, platform<Platform>(), Vec1D{extent});
    }
  }

  template <internal::concepts::bounded_array T, concepts::queue TQueue>
  host_buffer<T> make_host_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(host, queue, Vec1D{std::extent_v<T>});
    } else {
      using Platform = alpaka::Platform<alpaka::Dev<TQueue>>;
      return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(
          host, platform<Platform>(), Vec1D{std::extent_v<T>});
    }
  }

  // scalar and 1-dimensional host views

  template <typename T>
  using host_view = typename detail::view_type<DevHost, T>::type;

  template <internal::concepts::scalar T>
  host_view<T> make_host_view(T& data) {
    return alpaka::ViewPlainPtr<DevHost, T, Dim0D, Idx>(&data, host, Scalar{});
  }

  template <internal::concepts::scalar T>
  host_view<T[]> make_host_view(T* data, Extent extent) {
    return alpaka::ViewPlainPtr<DevHost, T, Dim1D, Idx>(data, host, Vec1D{extent});
  }

  template <internal::concepts::unbounded_array T>
  host_view<T> make_host_view(T& data, Extent extent) {
    return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(
        data, host, Vec1D{extent});
  }

  template <internal::concepts::bounded_array T>
  host_view<T> make_host_view(T& data) {
    return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(
        data, host, Vec1D{std::extent_v<T>});
  }

  // scalar and 1-dimensional device buffers

  template <typename TDev, typename T>
  using device_buffer = typename detail::buffer_type<TDev, T>::type;

  template <internal::concepts::scalar T, concepts::queue TQueue>
  device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<T, Idx>(alpaka::getDev(queue), queue, Scalar{});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<T, Idx>(queue, Scalar{});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<T, Idx>(alpaka::getDev(queue), Scalar{});
    }
  }

  template <internal::concepts::unbounded_array T, concepts::queue TQueue>
  device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue, Extent extent) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(
          alpaka::getDev(queue), queue, Vec1D{extent});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue, Vec1D{extent});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue), Vec1D{extent});
    }
  }

  template <internal::concepts::bounded_array T, concepts::queue TQueue>
  device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
      return allocCachedBuf<std::remove_extent_t<T>, Idx>(
          alpaka::getDev(queue), queue, Vec1D{std::extent_v<T>});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue, Vec1D{std::extent_v<T>});
    }
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue),
                                                            Vec1D{std::extent_v<T>});
    }
  }

  // scalar and 1-dimensional device views

  template <typename TDev, typename T>
  using device_view = typename detail::view_type<TDev, T>::type;

  template <internal::concepts::scalar T, concepts::device TDev>
  device_view<TDev, T> make_device_view(TDev const& device, T& data) {
    return alpaka::ViewPlainPtr<TDev, T, Dim0D, Idx>(&data, device, Scalar{});
  }

  template <internal::concepts::scalar T, concepts::device TDev>
  device_view<TDev, T[]> make_device_view(TDev const& device, T* data, Extent extent) {
    return alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>(data, device, Vec1D{extent});
  }

  template <internal::concepts::unbounded_array T, concepts::device TDev>
  device_view<TDev, T> make_device_view(TDev const& device, T& data, Extent extent) {
    return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(
        data, device, Vec1D{extent});
  }

  template <internal::concepts::bounded_array T, concepts::device TDev>
  device_view<TDev, T> make_device_view(TDev const& device, T& data) {
    return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(
        data, device, Vec1D{std::extent_v<T>});
  }

}  // namespace clue
