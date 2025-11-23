
#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace clue::internal::algorithm {

  template <typename InputIterator, typename Predicate>
  ALPAKA_FN_HOST inline constexpr auto count_if(InputIterator first,
                                                InputIterator last,
                                                Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::count_if(thrust::device, first, last, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::count_if(thrust::hip::par, first, last, pred);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::count_if(oneapi::dpl::execution::dpcpp_default, first, last, pred);
#else
    return std::count_if(first, last, pred);
#endif
  }

  template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
  ALPAKA_FN_HOST inline constexpr auto count_if(ExecutionPolicy&& policy,
                                                InputIterator first,
                                                InputIterator last,
                                                Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::count_if(std::forward<ExecutionPolicy>(policy), first, last, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::count_if(std::forward<ExecutionPolicy>(policy), first, last, pred);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::count_if(std::forward<ExecutionPolicy>(policy), first, last, pred);
#else
    return std::count_if(std::forward<ExecutionPolicy>(policy), first, last, pred);
#endif
  }

}  // namespace clue::internal::algorithm
