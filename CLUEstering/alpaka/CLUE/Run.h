
#pragma once

#include <vector>

#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <uint8_t Ndim, typename Kernel>
  std::vector<std::vector<int>> run(float dc,
                                    float rhoc,
                                    float dm,
                                    int pPBin,
                                    const std::vector<std::vector<float>>& coordinates,
                                    const std::vector<float>& weight,
                                    const Kernel& kernel,
                                    Queue queue_,
                                    size_t block_size) {
    CLUEAlgoAlpaka<Ndim> algo(dc, rhoc, dm, pPBin, queue_);

    // Create the host and device points
    Points<Ndim> h_points(coordinates, weight);
    PointsAlpaka<Ndim> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE
