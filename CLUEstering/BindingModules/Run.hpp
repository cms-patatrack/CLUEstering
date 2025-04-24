
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
           Queue queue_,
           size_t block_size) {
    CLUEAlgoAlpaka<Ndim> algo(dc, rhoc, dm, pPBin, queue_);

    // Create the host and device points
    PointsSoA<Ndim> h_points(std::get<0>(pData), std::get<1>(pData), n_points);
    clue::PointsAlpaka<Ndim, Device> d_points(queue_, n_points);

    algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
