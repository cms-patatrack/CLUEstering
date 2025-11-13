
#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace clue::internal::algorithm {

  template <typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr ForwardIterator min_element(ForwardIterator first,
                                                              ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(thrust::hip::par, first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::min_element(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::min_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr ForwardIterator min_element(ExecutionPolicy&& policy,
                                                              ForwardIterator first,
                                                              ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr ForwardIterator min_element(ForwardIterator first,
                                                              ForwardIterator last,
                                                              BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(thrust::hip::par, first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::min_element(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    return std::min_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr ForwardIterator min_element(ExecutionPolicy&& policy,
                                                              ForwardIterator first,
                                                              ForwardIterator last,
                                                              BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr ForwardIterator max_element(ForwardIterator first,
                                                              ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(thrust::hip::par, first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::max_element(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::max_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr ForwardIterator max_element(ExecutionPolicy&& policy,
                                                              ForwardIterator first,
                                                              ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr ForwardIterator max_element(ForwardIterator first,
                                                              ForwardIterator last,
                                                              BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(thrust::hip::par, first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::max_element(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    return std::max_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr ForwardIterator max_element(ExecutionPolicy&& policy,
                                                              ForwardIterator first,
                                                              ForwardIterator last,
                                                              BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ForwardIterator first, ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(thrust::hip::par, first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::minmax_element(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::minmax_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ForwardIterator first, ForwardIterator last, BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(thrust::hip::par, first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::minmax_element(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    return std::minmax_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last, BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) and not defined(ALPAKA_HOST_ONLY)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

}  // namespace clue::internal::algorithm
