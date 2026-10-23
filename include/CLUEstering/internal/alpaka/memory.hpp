
#pragma once

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/alpaka/config.hpp"

CLUE_NAMESPACE_BEGIN

// for Extent, Dim1D, Idx
using namespace alpaka_common;

// type deduction helpers
namespace detail {

  template <typename TDev, typename T>
  struct buffer_type {
    using type = alpaka::Buf<TDev, T, Dim0D, std::size_t>;
  };

  template <typename TDev, typename T>
  struct buffer_type<TDev, T[]> {
    using type = alpaka::Buf<TDev, T, Dim1D, std::size_t>;
  };

  template <typename TDev, typename T, int N>
  struct buffer_type<TDev, T[N]> {
    using type = alpaka::Buf<TDev, T, Dim1D, std::size_t>;
  };

}  // namespace detail

template <typename T>
using host_buffer = typename detail::buffer_type<DevHost, T>::type;

template <typename TDev, typename T>
using device_buffer = typename detail::buffer_type<TDev, T>::type;

CLUE_NAMESPACE_END
