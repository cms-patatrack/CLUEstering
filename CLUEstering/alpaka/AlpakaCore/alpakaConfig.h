
#pragma once

#include "alpakaFwd.h"

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
  using PltfHost = alpaka::PltfCpu;

}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_PRESENT
namespace alpaka_cuda_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCudaRt;
  using Device = alpaka::DevCudaRt;
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using Event = alpaka::EventCudaRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuCudaRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_PRESENT

#ifdef ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda_async
#endif  // ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND

#ifdef ALPAKA_ACC_GPU_HIP_PRESENT
namespace alpaka_rocm_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfHipRt;
  using Device = alpaka::DevHipRt;
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Event = alpaka::EventHipRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuHipRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_rocm_async

#endif  // ALPAKA_ACC_GPU_HIP_PRESENT

#ifdef ALPAKA_ACC_GPU_HIP_ASYNC_BACKEND
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_rocm_async
#endif  // ALPAKA_ACC_GPU_HIP_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
namespace alpaka_serial_sync {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuSerial<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT
namespace alpaka_tbb_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuNonBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_tbb_async
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_PRESENT
namespace alpaka_omp2_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuOmp2Blocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_omp2_async

#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_PRESENT

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp2_async
#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
