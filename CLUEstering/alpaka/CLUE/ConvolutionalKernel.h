#ifndef convolutional_kernels_h
#define convolutional_kernels_h

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Debug.hpp>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

using kernel_t = std::function<float(float, int, int)>;

struct ConvolutionalKernel {
private:
  kernel_t m_function;

public:
  // Constructors
  ConvolutionalKernel() = delete;
  ConvolutionalKernel(kernel_t function) : m_function{std::move(function)} {}

  // Overload call operator
  ALPAKA_FN_HOST_ACC float operator()(float dist_ij, int point_id, int j) const {
    return m_function(dist_ij, point_id, j);
  }
};

#endif
