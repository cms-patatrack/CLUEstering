
#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace clue::internal::algorithm {

  template <typename InputIterator>
  ALPAKA_FN_HOST inline constexpr typename std::iterator_traits<InputIterator>::value_type reduce(
      InputIterator first, InputIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(thrust::hip::par, first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::reduce(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::reduce(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr typename std::iterator_traits<ForwardIterator>::value_type reduce(
      ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename InputIterator, typename T>
  ALPAKA_FN_HOST inline constexpr T reduce(InputIterator first, InputIterator last, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(thrust::device, first, last, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(thrust::hip::par, first, last, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::reduce(oneapi::dpl::execution::dpcpp_default, first, last, init);
#else
    return std::reduce(first, last, init);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
  ALPAKA_FN_HOST inline constexpr T reduce(ExecutionPolicy&& policy,
                                           ForwardIterator first,
                                           ForwardIterator last,
                                           T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#endif
  }

  template <typename InputIterator, typename T, typename BinaryOperation>
  ALPAKA_FN_HOST inline constexpr T reduce(InputIterator first,
                                           InputIterator last,
                                           T init,
                                           BinaryOperation op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(thrust::device, first, last, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(thrust::hip::par, first, last, init, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::reduce(oneapi::dpl::execution::dpcpp_default, first, last, init, op);
#else
    return std::reduce(first, last, init, op);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryOperation>
  ALPAKA_FN_HOST inline constexpr T reduce(ExecutionPolicy&& policy,
                                           ForwardIterator first,
                                           ForwardIterator last,
                                           T init,
                                           BinaryOperation op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#endif
  }

}  // namespace clue::internal::algorithm
