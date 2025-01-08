
#pragma once

#include <algorithm>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "alpakaConfig.h"

using namespace alpaka_common;

namespace clue {

  /*********************************************
   *              WORKDIV CREATION
   ********************************************/

  /*
   * If the first argument is not a multiple of the second argument, round it up to the next multiple.
   */
  inline constexpr Idx round_up_by(Idx value, Idx divisor) {
    return (value + divisor - 1) / divisor * divisor;
  }

  /*
   * Return the integer division of the first argument by the second argument, rounded up to the next integer.
   */
  inline constexpr Idx divide_up_by(Idx value, Idx divisor) {
    return (value + divisor - 1) / divisor;
  }

  /*
   * Creates the accelerator-dependent workdiv for 1-dimensional operations.
   */
  template <typename TAcc>
  inline WorkDiv<Dim1D> make_workdiv(Idx blocksPerGrid,
                                     Idx threadsPerBlockOrElementsPerThread) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TAcc, alpaka::AccGpuCudaRt<Dim1D, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - threadsPerBlockOrElementsPerThread is the number of threads per block;
      //   - elementsPerThread is always 1.
      const auto elementsPerThread = Idx{1};
      return WorkDiv<Dim1D>(
          blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
    } else
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#if ALPAKA_ACC_GPU_HIP_ENABLED
        if constexpr (std::is_same_v<TAcc, alpaka::AccGpuHipRt<Dim1D, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - threadsPerBlockOrElementsPerThread is the number of threads per block;
      //   - elementsPerThread is always 1.
      const auto elementsPerThread = Idx{1};
      return WorkDiv<Dim1D>(
          blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
    } else
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    {
      // On CPU backends, run serially with a single thread per block:
      //   - threadsPerBlock is always 1;
      //   - threadsPerBlockOrElementsPerThread is the number of elements per thread.
      const auto threadsPerBlock = Idx{1};
      return WorkDiv<Dim1D>(
          blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
    }
  }

  /*
   * Creates the accelerator-dependent workdiv for N-dimensional operations.
   */
  template <typename TAcc>
  inline WorkDiv<alpaka::Dim<TAcc>> make_workdiv(
      const Vec<alpaka::Dim<TAcc>>& blocksPerGrid,
      const Vec<alpaka::Dim<TAcc>>& threadsPerBlockOrElementsPerThread) {
    using Dim = alpaka::Dim<TAcc>;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TAcc, alpaka::AccGpuCudaRt<Dim, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - threadsPerBlockOrElementsPerThread is the number of threads per block;
      //   - elementsPerThread is always 1.
      const auto elementsPerThread = Vec<Dim>::ones();
      return WorkDiv<Dim>(
          blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
    } else
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        if constexpr (std::is_same_v<TAcc, alpaka::AccGpuHipRt<Dim, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - threadsPerBlockOrElementsPerThread is the number of threads per block;
      //   - elementsPerThread is always 1.
      const auto elementsPerThread = Vec<Dim>::ones();
      return WorkDiv<Dim>(
          blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
    } else
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    {
      // On CPU backends, run serially with a single thread per block:
      //   - threadsPerBlock is always 1;
      //   - threadsPerBlockOrElementsPerThread is the number of elements per thread.
      const auto threadsPerBlock = Vec<Dim>::ones();
      return WorkDiv<Dim>(
          blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
    }
  }

  /*********************************************
   *           RANGE COMPUTATION
   ********************************************/

  /*
   * Computes the range of the elements indexes, local to the block.
   * Warning: the max index is not truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block(
      const TAcc& acc, const Idx elementIdxShift, const unsigned int dimIndex = 0u) {
    // Take into account the thread index in block.
    const Idx threadIdxLocal(
        alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex]);
    const Idx threadDimension(
        alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

    // Compute the elements indexes in block.
    // Obviously relevant for CPU only.
    // For GPU, threadDimension == 1, and elementIdx == firstElementIdx == threadIdx + elementIdxShift.
    const Idx firstElementIdxLocal = threadIdxLocal * threadDimension;
    const Idx firstElementIdx = firstElementIdxLocal + elementIdxShift;  // Add the shift!
    const Idx endElementIdxUncut = firstElementIdx + threadDimension;

    // Return element indexes, shifted by elementIdxShift.
    return {firstElementIdx, endElementIdxUncut};
  }

  /*
   * Computes the range of the elements indexes, local to the block.
   * Truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block_truncated(
      const TAcc& acc,
      const Idx maxNumberOfElements,
      const Idx elementIdxShift,
      const unsigned int dimIndex = 0u) {
    // Check dimension
    //static_assert(alpaka::Dim<TAcc>::value == Dim1D::value,
    //              "Accelerator and maxNumberOfElements need to have same dimension.");
    auto [firstElementIdxLocal, endElementIdxLocal] =
        element_index_range_in_block(acc, elementIdxShift, dimIndex);

    // Truncate
    endElementIdxLocal = std::min(endElementIdxLocal, maxNumberOfElements);

    // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
    return {firstElementIdxLocal, endElementIdxLocal};
  }

  /*
   * Computes the range of the elements indexes in grid.
   * Warning: the max index is not truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid(
      const TAcc& acc, Idx elementIdxShift, const unsigned int dimIndex = 0u) {
    // Take into account the block index in grid.
    const Idx blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
    const Idx blockDimension(
        alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

    // Shift to get global indices in grid (instead of local to the block)
    elementIdxShift += blockIdxInGrid * blockDimension;

    // Return element indexes, shifted by elementIdxShift.
    return element_index_range_in_block(acc, elementIdxShift, dimIndex);
  }

  /*
   * Computes the range of the elements indexes in grid.
   * Truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid_truncated(
      const TAcc& acc,
      const Idx maxNumberOfElements,
      Idx elementIdxShift,
      const unsigned int dimIndex = 0u) {
    // Check dimension
    //static_assert(dimIndex <= alpaka::Dim<TAcc>::value,
    //"Accelerator and maxNumberOfElements need to have same dimension.");
    auto [firstElementIdxGlobal, endElementIdxGlobal] =
        element_index_range_in_grid(acc, elementIdxShift, dimIndex);

    // Truncate
    endElementIdxGlobal = std::min(endElementIdxGlobal, maxNumberOfElements);

    // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
    return {firstElementIdxGlobal, endElementIdxGlobal};
  }

  /*
   * Computes the range of the element(s) index(es) in grid.
   * Truncated by the max number of elements of interest.
   */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid_truncated(
      const TAcc& acc, const Idx maxNumberOfElements, const unsigned int dimIndex = 0u) {
    Idx elementIdxShift = 0u;
    return element_index_range_in_grid_truncated(
        acc, maxNumberOfElements, elementIdxShift, dimIndex);
  }

  /*********************************************
   *           LOOP ON ALL ELEMENTS
   ********************************************/

  /*
   * Loop on all (CPU) elements.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Indexes are local to the BLOCK.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block(const TAcc& acc,
                                               const Idx maxNumberOfElements,
                                               const Idx elementIdxShift,
                                               const Func func,
                                               const unsigned int dimIndex = 0) {
    const auto& [firstElementIdx, endElementIdx] = element_index_range_in_block_truncated(
        acc, maxNumberOfElements, elementIdxShift, dimIndex);

    for (Idx elementIdx = firstElementIdx; elementIdx < endElementIdx; ++elementIdx) {
      func(elementIdx);
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block(const TAcc& acc,
                                               const Idx maxNumberOfElements,
                                               const Func func,
                                               const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_block(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /*
   * Loop on all (CPU) elements.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Indexes are expressed in GRID 'frame-of-reference'.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid(const TAcc& acc,
                                              const Idx maxNumberOfElements,
                                              Idx elementIdxShift,
                                              const Func func,
                                              const unsigned int dimIndex = 0) {
    // Take into account the block index in grid to compute the element indices.
    const Idx blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
    const Idx blockDimension(
        alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);
    elementIdxShift += blockIdxInGrid * blockDimension;

    for_each_element_in_block(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid(const TAcc& acc,
                                              const Idx maxNumberOfElements,
                                              const Func func,
                                              const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_grid(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /**************************************************************
   *          LOOP ON ALL ELEMENTS, WITH STRIDED ACCESS
   **************************************************************/

  /*
   * (CPU) Loop on all elements + (CPU/GPU) Strided access.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Stride to full problem size, by BLOCK size.
   * Indexes are local to the BLOCK.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block_strided(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Idx elementIdxShift,
                                                       const Func func,
                                                       const unsigned int dimIndex = 0) {
    // Get thread / element indices in block.
    const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
        element_index_range_in_block(acc, elementIdxShift, dimIndex);

    // Stride = block size.
    const Idx blockDimension(
        alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

    // Strided access.
    for (Idx threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
         threadIdx < maxNumberOfElements;
         threadIdx += blockDimension, endElementIdx += blockDimension) {
      // (CPU) Loop on all elements.
      if (endElementIdx > maxNumberOfElements) {
        endElementIdx = maxNumberOfElements;
      }
      for (Idx i = threadIdx; i < endElementIdx; ++i) {
        func(i);
      }
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block_strided(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Func func,
                                                       const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_block_strided(
        acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /*
   * (CPU) Loop on all elements + (CPU/GPU) Strided access.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Stride to full problem size, by GRID size.
   * Indexes are local to the GRID.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid_strided(const TAcc& acc,
                                                      const Idx maxNumberOfElements,
                                                      const Idx elementIdxShift,
                                                      const Func func,
                                                      const unsigned int dimIndex = 0) {
    // Get thread / element indices in block.
    const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
        element_index_range_in_grid(acc, elementIdxShift, dimIndex);

    // Stride = grid size.
    const Idx gridDimension(
        alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[dimIndex]);

    // Strided access.
    for (Idx threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
         threadIdx < maxNumberOfElements;
         threadIdx += gridDimension, endElementIdx += gridDimension) {
      // (CPU) Loop on all elements.
      if (endElementIdx > maxNumberOfElements) {
        endElementIdx = maxNumberOfElements;
      }
      for (Idx i = threadIdx; i < endElementIdx; ++i) {
        func(i);
      }
    }
  }

  /* once_per_grid
   *
   * `once_per_grid(acc)` returns true for a single thread within the kernel execution grid.
   *
   * Usually the condition is true for block 0 and thread 0, but these indices should not be relied upon.
   */
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC inline constexpr bool once_per_grid(TAcc const& acc) {
    return alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) ==
           Vec<alpaka::Dim<TAcc>>::zeros();
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid_strided(const TAcc& acc,
                                                      const Idx maxNumberOfElements,
                                                      const Func func,
                                                      const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_grid_strided(
        acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /**************************************************************
   *          LOOP ON ALL ELEMENTS WITH ONE LOOP
   **************************************************************/

  /*
   * Case where the input index i has reached the end of threadDimension: strides the input index.
   * Otherwise: do nothing.
   * NB 1: This helper function is used as a trick to only have one loop (like in legacy), instead of 2 loops
   * (like in all the other Alpaka helpers, 'for_each_element_in_block_strided' for example, 
   * because of the additional loop over elements in Alpaka model). 
   * This allows to keep the 'continue' and 'break' statements as-is from legacy code, 
   * and hence avoids a lot of legacy code reshuffling.
   * NB 2: Modifies i, firstElementIdx and endElementIdx.
   */
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool next_valid_element_index_strided(
      Idx& i,
      Idx& firstElementIdx,
      Idx& endElementIdx,
      const Idx stride,
      const Idx maxNumberOfElements) {
    bool isNextStrideElementValid = true;
    if (i == endElementIdx) {
      firstElementIdx += stride;
      endElementIdx += stride;
      i = firstElementIdx;
      if (i >= maxNumberOfElements) {
        isNextStrideElementValid = false;
      }
    }
    return isNextStrideElementValid;
  }

}  // namespace clue
