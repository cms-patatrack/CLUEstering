
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
           const PointInfo<Ndim>& shape,
           const Kernel& kernel,
           Queue queue_,
           size_t block_size) {
    CLUEAlgoAlpaka<Ndim> algo(dc, rhoc, dm, pPBin, queue_);

    // Create the host and device points
    PointsSoA<Ndim> h_points(std::get<0>(pData), std::get<1>(pData), shape);
    clue::PointsAlpaka<Ndim, Device> d_points(queue_, shape.nPoints);

    algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
