#ifndef run_h
#define run_h

#include <vector>
#include "CLUEAlgoAlpaka.h"
#include "ConvolutionalKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  std::vector<std::vector<int>> run1(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run1(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run1(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run2(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run2(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run2(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run3(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run3(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run3(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run4(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run4(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run4(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run5(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run5(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run5(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run6(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run6(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run6(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run7(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run7(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run7(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run8(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run8(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run8(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run9(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const FlatKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run9(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const ExponentialKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run9(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     const GaussianKernel &kernel,
                                     Queue queue_,
                                     size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run10(float dc,
                                      float rhoc,
                                      float outlier,
                                      int pPBin,
                                      std::vector<std::vector<float>> const &coordinates,
                                      std::vector<float> const &weight,
                                      const FlatKernel &kernel,
                                      Queue queue_,
                                      size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run10(float dc,
                                      float rhoc,
                                      float outlier,
                                      int pPBin,
                                      std::vector<std::vector<float>> const &coordinates,
                                      std::vector<float> const &weight,
                                      const ExponentialKernel &kernel,
                                      Queue queue_,
                                      size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

  std::vector<std::vector<int>> run10(float dc,
                                      float rhoc,
                                      float outlier,
                                      int pPBin,
                                      std::vector<std::vector<float>> const &coordinates,
                                      std::vector<float> const &weight,
                                      const GaussianKernel &kernel,
                                      Queue queue_,
                                      size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

    // Create the host and device points
    Points<2> h_points(coordinates, weight);
    PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
