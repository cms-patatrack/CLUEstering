#ifndef run_h
#define run_h

#include <vector>
#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  std::vector<std::vector<int>> run1(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
                                     Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 1> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<1> h_points(coordinates, weight);
	PointsAlpaka<1> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run2(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 2> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<2> h_points(coordinates, weight);
	PointsAlpaka<2> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run3(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 3> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<3> h_points(coordinates, weight);
	PointsAlpaka<3> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run4(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 4> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<4> h_points(coordinates, weight);
	PointsAlpaka<4> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run5(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 5> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<5> h_points(coordinates, weight);
	PointsAlpaka<5> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run6(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 6> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<6> h_points(coordinates, weight);
	PointsAlpaka<6> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run7(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 7> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<7> h_points(coordinates, weight);
	PointsAlpaka<7> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run8(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 8> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<8> h_points(coordinates, weight);
	PointsAlpaka<8> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run9(float dc,
                                     float rhoc,
                                     float outlier,
                                     int pPBin,
                                     std::vector<std::vector<float>> const &coordinates,
                                     std::vector<float> const &weight,
									 Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 9> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<9> h_points(coordinates, weight);
	PointsAlpaka<9> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }

  std::vector<std::vector<int>> run10(float dc,
                                      float rhoc,
                                      float outlier,
                                      int pPBin,
                                      std::vector<std::vector<float>> const &coordinates,
                                      std::vector<float> const &weight,
									  Queue queue_) {
    CLUEAlgoAlpaka<Acc1D, 10> algo(dc, rhoc, outlier, pPBin, queue_);

	// Create the host and device points
	Points<10> h_points(coordinates, weight);
	PointsAlpaka<10> d_points(queue_, weight.size());

    return algo.make_clusters(h_points, d_points, queue_);
  }
};  // namespace alpaka_serial_sync

#endif
