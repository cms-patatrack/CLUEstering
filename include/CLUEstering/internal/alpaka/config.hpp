
#pragma once

#include <alpaka/alpaka.hpp>

namespace clue::alpaka_common {

  // common types and dimensions
  using Idx = uint32_t;
  using Extent = uint32_t;
  using Offsets = Extent;

  using Dim0D = alpaka::DimInt<0u>;
  using Dim1D = alpaka::DimInt<1u>;
  using Dim2D = alpaka::DimInt<2u>;
  using Dim3D = alpaka::DimInt<3u>;

  template <typename TDim>
  using Vec = alpaka::Vec<TDim, Idx>;
  using Vec1D = Vec<Dim1D>;
  using Vec2D = Vec<Dim2D>;
  using Vec3D = Vec<Dim3D>;
  using Scalar = alpaka::Vec<Dim0D, std::size_t>;

  template <typename TDim>
  using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
  using WorkDiv1D = WorkDiv<Dim1D>;
  using WorkDiv2D = WorkDiv<Dim2D>;
  using WorkDiv3D = WorkDiv<Dim3D>;

  // host types
  using DevHost = alpaka::DevCpu;
  using PlatformHost = alpaka::PlatformCpu;

}  // namespace clue::alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
namespace clue::alpaka_cuda_async {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformCudaRt;
  using Device = alpaka::DevCudaRt;
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using Event = alpaka::EventCudaRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuCudaRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
namespace clue::alpaka_rocm_async {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformHipRt;
  using Device = alpaka::DevHipRt;
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Event = alpaka::EventHipRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuHipRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_rocm_async
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
#ifdef ALPAKA_SYCL_ONEAPI_CPU
namespace clue::alpaka_sycl_cpu {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformCpuSycl;
  using Device = alpaka::DevCpuSycl;
  using Queue = alpaka::QueueCpuSyclBlocking;
  using Event = alpaka::EventCpuSycl;

  template <typename TDim>
  using Acc = alpaka::AccCpuSycl<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_sycl_cpu

#elif ALPAKA_SYCL_ONEAPI_GPU
namespace clue::alpaka_sycl_gpu {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformGpuSyclIntel;
  using Device = alpaka::DevGpuSyclIntel;
  using Queue = alpaka::QueueGpuSyclIntelNonBlocking;
  using Event = alpaka::EventGpuSyclIntel;

  template <typename TDim>
  using Acc = alpaka::AccGpuSyclIntel<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_sycl_gpu

#endif
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
namespace clue::alpaka_serial_sync {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuSerial<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
namespace clue::alpaka_tbb_async {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuNonBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
namespace clue::alpaka_omp2_sync {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuOmp2Blocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_omp2_sync

#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
namespace clue::alpaka_threads_sync {
  using namespace alpaka_common;

  using Platform = alpaka::PlatformCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuThreads<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace clue::alpaka_threads_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

// Set ALPAKA_BACKEND to the highest-priority enabled backend exactly once.
// Alpaka always enables ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED alongside every
// other CPU backend, so using a single elif chain avoids redefinition warnings.
// CLUE_ABI_TAG names the backend-specific inline namespace opened by CLUE_NAMESPACE_BEGIN,
// so translation units built for different backends get distinct symbol names and can be
// linked into one program without ODR clashes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#define ALPAKA_BACKEND clue::alpaka_cuda_async
#define CLUE_ABI_TAG backend_cuda
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define ALPAKA_BACKEND clue::alpaka_rocm_async
#define CLUE_ABI_TAG backend_hip
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)
#define ALPAKA_BACKEND clue::alpaka_sycl_gpu
#define CLUE_ABI_TAG backend_sycl_gpu
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)
#define ALPAKA_BACKEND clue::alpaka_sycl_cpu
#define CLUE_ABI_TAG backend_sycl_cpu
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#define ALPAKA_BACKEND clue::alpaka_tbb_async
#define CLUE_ABI_TAG backend_tbb
#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#define ALPAKA_BACKEND clue::alpaka_omp2_sync
#define CLUE_ABI_TAG backend_omp2
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
#define ALPAKA_BACKEND clue::alpaka_threads_sync
#define CLUE_ABI_TAG backend_threads
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
#define ALPAKA_BACKEND clue::alpaka_serial_sync
#define CLUE_ABI_TAG backend_serial
#endif

#ifndef CLUE_ABI_TAG
#error "CLUEstering: no alpaka backend enabled, define one ALPAKA_ACC_* macro"
#endif

// Open and close the CLUEstering namespace. All the code that depends on the backend must
// be declared between these two macros, while users can keep referring to it as clue::
#define CLUE_NAMESPACE_BEGIN \
  namespace clue {           \
    inline namespace CLUE_ABI_TAG {
#define CLUE_NAMESPACE_END \
  }                        \
  }
