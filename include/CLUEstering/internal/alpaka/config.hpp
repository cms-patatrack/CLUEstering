
#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka_common {

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
  using Scalar = Vec<Dim0D>;

  template <typename TDim>
  using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
  using WorkDiv1D = WorkDiv<Dim1D>;
  using WorkDiv2D = WorkDiv<Dim2D>;
  using WorkDiv3D = WorkDiv<Dim3D>;

  // host types
  using DevHost = alpaka::DevCpu;
  using PlatformHost = alpaka::PlatformCpu;

}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
namespace alpaka_cuda_async {
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

}  // namespace alpaka_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
namespace alpaka_rocm_async {
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

}  // namespace alpaka_rocm_async
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
#ifdef ALPAKA_SYCL_ONEAPI_CPU
namespace alpaka_sycl_cpu {
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

}  // namespace alpaka_sycl_cpu

#elif ALPAKA_SYCL_ONEAPI_GPU
namespace alpaka_sycl_gpu {
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

}  // namespace alpaka_sycl_gpu

#endif
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
namespace alpaka_serial_sync {
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

}  // namespace alpaka_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
namespace alpaka_tbb_async {
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

}  // namespace alpaka_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
namespace alpaka_omp2_sync {
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

}  // namespace alpaka_omp2_sync

#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
namespace alpaka_threads_sync {
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

}  // namespace alpaka_threads_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

// Set ALPAKA_BACKEND to the highest-priority enabled backend exactly once.
// Alpaka always enables ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED alongside every
// other CPU backend, so using a single elif chain avoids redefinition warnings.
// CLUE_BACKEND names the inline namespace (clue::inline CLUE_BACKEND) that every
// CLUEstering header opens, so translation units built for different backends get
// distinct symbol names and can be linked into one program without ODR clashes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#define ALPAKA_BACKEND alpaka_cuda_async
#define CLUE_BACKEND cuda
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define ALPAKA_BACKEND alpaka_rocm_async
#define CLUE_BACKEND hip
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)
#define ALPAKA_BACKEND alpaka_sycl_gpu
#define CLUE_BACKEND sycl_gpu
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)
#define ALPAKA_BACKEND alpaka_sycl_cpu
#define CLUE_BACKEND sycl_cpu
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#define ALPAKA_BACKEND alpaka_tbb_async
#define CLUE_BACKEND tbb
#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#define ALPAKA_BACKEND alpaka_omp2_sync
#define CLUE_BACKEND omp2
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
#define ALPAKA_BACKEND alpaka_threads_sync
#define CLUE_BACKEND threads
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
#define ALPAKA_BACKEND alpaka_serial_sync
#define CLUE_BACKEND serial
#endif

#ifndef CLUE_BACKEND
#error "CLUEstering: no alpaka backend enabled, define one ALPAKA_ACC_* macro"
#endif
