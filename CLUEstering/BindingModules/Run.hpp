
#pragma once

#include <vector>
#include "CLUEstering/CLUEstering.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  template <uint8_t Ndim, typename Kernel>
  void run(float dc,
           float rhoc,
           float dm,
           int pPBin,
           std::tuple<float*, int*>&& pData,
           uint32_t n_points,
           const Kernel& kernel,
           Queue queue,
           size_t block_size) {
    CLUEAlgoAlpaka<Ndim> algo(dc, rhoc, dm, pPBin, queue);

    // Create the host and device points
    clue::PointsHost<Ndim> h_points(
        queue, n_points, std::get<0>(pData), std::get<1>(pData));
    clue::PointsDevice<Ndim, Device> d_points(queue, n_points);

    algo.make_clusters(h_points, d_points, kernel, queue, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
