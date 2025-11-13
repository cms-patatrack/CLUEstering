

#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace clue::internal::algorithm {

  template <typename RandomAccessIterator>
  ALPAKA_FN_HOST inline constexpr void sort(RandomAccessIterator first, RandomAccessIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(thrust::hip::par, first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    std::sort(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator>
  ALPAKA_FN_HOST inline constexpr void sort(ExecutionPolicy&& policy,
                                            RandomAccessIterator first,
                                            RandomAccessIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::sort(std::forward<ExecutionPolicy>(policy), first, last);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename RandomAccessIterator, typename Compare>
  ALPAKA_FN_HOST inline constexpr void sort(RandomAccessIterator first,
                                            RandomAccessIterator last,
                                            Compare comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(thrust::hip::par, first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    std::sort(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
  ALPAKA_FN_HOST inline constexpr void sort(ExecutionPolicy&& policy,
                                            RandomAccessIterator first,
                                            RandomAccessIterator last,
                                            Compare comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

}  // namespace clue::internal::algorithm
