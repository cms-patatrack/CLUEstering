
#pragma once

#include "CLUEstering/detail/concepts.hpp"

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace clue::internal::algorithm {

  template <typename InputIterator, typename OutputIterator>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator output) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device, first, last, output);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par, first, last, output);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, output);
#else
    std::inclusive_scan(first, last, output);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                                      ForwardIterator first,
                                                      ForwardIterator last,
                                                      ForwardIterator output) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output);
#else
    std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename BinaryOperator>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator output,
                                                      BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device, first, last, output, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par, first, last, output, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, output, op);
#else
    std::inclusive_scan(first, last, output, op);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryOperator>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                                      ForwardIterator first,
                                                      ForwardIterator last,
                                                      ForwardIterator output,
                                                      BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op);
#else
    std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename BinaryOperator, typename T>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(
      InputIterator first, InputIterator last, OutputIterator output, BinaryOperator op, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device, first, last, output, op, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par, first, last, output, op, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::dpcpp_default, first, last, output, op, init);
#else
    std::inclusive_scan(first, last, output, op, init);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryOperator, typename T>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                                      ForwardIterator first,
                                                      ForwardIterator last,
                                                      ForwardIterator output,
                                                      BinaryOperator op,
                                                      T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(
        std::forward<ExecutionPolicy>(policy), first, last, output, op, init);
#else
    std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op, init);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename T>
  ALPAKA_FN_HOST inline constexpr void exclusive_scan(InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator output,
                                                      T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(thrust::device, first, last, output, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(thrust::hip::par, first, last, output, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::exclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, output, init);
#else
    std::exclusive_scan(first, last, output, init);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
  ALPAKA_FN_HOST inline constexpr void exclusive_scan(ExecutionPolicy&& policy,
                                                      ForwardIterator first,
                                                      ForwardIterator last,
                                                      ForwardIterator output,
                                                      T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init);
#else
    std::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOperator>
  ALPAKA_FN_HOST inline constexpr void exclusive_scan(
      InputIterator first, InputIterator last, OutputIterator output, T init, BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(thrust::device, first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(thrust::hip::par, first, last, output, init, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::exclusive_scan(
        oneapi::dpl::execution::dpcpp_default, first, last, output, init, op);
#else
    std::exclusive_scan(first, last, output, init, op);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryOperator>
  ALPAKA_FN_HOST inline constexpr void exclusive_scan(ExecutionPolicy&& policy,
                                                      ForwardIterator first,
                                                      ForwardIterator last,
                                                      ForwardIterator output,
                                                      T init,
                                                      BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::exclusive_scan(
        std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#else
    std::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(TQueue& queue,
                                                      InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator output) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device.on(queue.getNativeHandle()), first, last, output);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::device.on(queue.getNativeHandle()), first, last, output);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, output);
#else
    std::inclusive_scan(first, last, output);
#endif
  }

  template <concepts::queue TQueue,
            typename InputIterator,
            typename OutputIterator,
            typename BinaryOperator>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(TQueue& queue,
                                                      InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator output,
                                                      BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device.on(queue.getNativeHandle()), first, last, output, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::device.on(queue.getNativeHandle()), first, last, output, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, output, op);
#else
    std::inclusive_scan(first, last, output, op);
#endif
  }

  template <concepts::queue TQueue,
            typename InputIterator,
            typename OutputIterator,
            typename BinaryOperator,
            typename T>
  ALPAKA_FN_HOST inline constexpr void inclusive_scan(TQueue& queue,
                                                      InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator output,
                                                      BinaryOperator op,
                                                      T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(
        thrust::device.on(queue.getNativeHandle()), first, last, output, op, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(
        thrust::device.on(queue.getNativeHandle()), first, last, output, op, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::dpcpp_default, first, last, output, op, init);
#else
    std::inclusive_scan(first, last, output, op, init);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator, typename T>
  ALPAKA_FN_HOST inline constexpr void exclusive_scan(
      TQueue& queue, InputIterator first, InputIterator last, OutputIterator output, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(thrust::device.on(queue.getNativeHandle()), first, last, output, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(thrust::device.on(queue.getNativeHandle()), first, last, output, init);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::exclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, output, init);
#else
    std::exclusive_scan(first, last, output, init);
#endif
  }

  template <concepts::queue TQueue,
            typename InputIterator,
            typename OutputIterator,
            typename T,
            typename BinaryOperator>
  ALPAKA_FN_HOST inline constexpr void exclusive_scan(TQueue& queue,
                                                      InputIterator first,
                                                      InputIterator last,
                                                      OutputIterator output,
                                                      T init,
                                                      BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(
        thrust::device.on(queue.getNativeHandle()), first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(
        thrust::device.on(queue.getNativeHandle()), first, last, output, init, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::exclusive_scan(
        oneapi::dpl::execution::dpcpp_default, first, last, output, init, op);
#else
    std::exclusive_scan(first, last, output, init, op);
#endif
  }

}  // namespace clue::internal::algorithm
