
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include <alpaka/alpaka.hpp>

CLUE_NAMESPACE_BEGIN

// generic interface, for DevOacc and DevOmp5
template <typename Device>
inline int getDeviceIndex(const Device& device) {
  return device.iDevice();
}

// overload for DevCpu
inline int getDeviceIndex(const alpaka::DevCpu&) { return 0; }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
// overload for DevCudaRt
inline int getDeviceIndex(alpaka::DevCudaRt const& device) {
  return alpaka::getNativeHandle(device);
}
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
// overload for DevHipRt
inline int getDeviceIndex(alpaka::DevHipRt const& device) {
  return alpaka::getNativeHandle(device);
}
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

CLUE_NAMESPACE_END
