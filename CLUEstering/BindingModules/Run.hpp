
#pragma once

#include <vector>
#include "CLUEstering/CLUEstering.hpp"

template <uint8_t Ndim, clue::concepts::convolutional_kernel Kernel>
void run(float dc,
         float rhoc,
         float dm,
         float seed_dc,
         int pPBin,
         std::tuple<float*, int*>&& pData,
         int32_t n_points,
         const Kernel& kernel,
         clue::Queue queue,
         size_t block_size) {
  clue::Clusterer<Ndim> algo(queue, dc, rhoc, dm, seed_dc, pPBin);

  // Create the host and device points
  clue::PointsHost<Ndim> h_points(queue, n_points, std::get<0>(pData), std::get<1>(pData));
  clue::PointsDevice<Ndim> d_points(queue, n_points);

  algo.make_clusters(queue, h_points, d_points, kernel, block_size);
}
