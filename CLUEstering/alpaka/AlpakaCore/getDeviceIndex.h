#ifndef AlpakaCore_getDeviceIndex_h
#define AlpakaCore_getDeviceIndex_h

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  // generic interface, for DevOacc and DevOmp5
  template <typename Device>
  inline int getDeviceIndex(Device const& device) {
    return device.iDevice();
  }

  // overload for DevCpu
  inline int getDeviceIndex(alpaka::DevCpu const& device) { return 0; }

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

}  // namespace cms::alpakatools

#endif  // AlpakaCore_getDeviceIndex_h
