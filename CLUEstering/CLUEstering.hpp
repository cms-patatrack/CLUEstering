
#ifndef cluestering_hpp
#define cluestering_hpp

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "DataFormats/Points.h"
#include "DataFormats/alpaka/PointsAlpaka.h"
#include "CLUE/CLUEAlgoAlpaka.h"
#include "CLUE/Run.h"

using Dim = uint8_t;

template <Dim ndim>
class Clusterer {
private:
  float m_dc, m_rhoc, m_outlier;
  uint32_t m_pointePerTile;
  // Points SoA (Structure of Array)
  Points<ndim> m_points;
  /* std::unique_ptr<ConvolutionalKernel> m_kernel; */		// todo when kernels reworked
  std::vector<std::vector<int>> m_clusterResults;

public:
  Clusterer() = delete;
  Clusterer(float dc, float rhoc, float outlier, uint32_t pointsPerTile = 128)
      : m_dc{dc}, m_rhoc{rhoc}, m_outlier{outlier}, m_pointePerTile{pointsPerTile} {}

  template <typename Vec, typename... Vecs>		// todo: need to add a constraint on the template parameters
  void read_data(Vec vec, Vecs... vecs) {
    if constexpr (sizeof...(vecs) == 0) {
      m_points.m_weight = vec;
    } else {
      std::for_each(vec.begin(), vec.end(), [&, this](auto& x) {
        /* m_points.m_coords.push_back(VecArray<float, ndim>{x, vecs...}); */
      });
	  read_data(std::forward(vecs)...);
    }
  }

  void run_clue(const std::string& backend,
                std::size_t block_size = 256,
                std::size_t device_id = 0,
                bool verbose = false) {
	if (backend == "cpu serial") {
	  using namespace alpaka_serial_sync;
	  const auto device = alpaka::getDevByIdx<Acc1D>(device_id);
	  Queue queue(device);

	  // note: alternatively just call run<Ndim, Kernel>
	  CLUEAlgoAlpaka<Acc1D, ndim> algo(m_dc, m_rhoc, m_outlier, m_pointePerTile, queue);
	  PointsAlpaka<ndim> d_points(queue, m_points.n);

	  algo.make_clusters(m_points, d_points, ConvolutionalKernel(), queue, block_size);
	} else if (backend == "cpu tbb") {
	  using namespace alpaka_tbb_async;
	} else if (backend == "gpu cuda") {
	  using namespace alpaka_cuda_async;
	} else if (backend == "gpu hip") {
	  using namespace alpaka_rocm_async;
	}
  }

  void to_csv(const std::string& pathToFile) {
    std::fstream csvFile(pathToFile);
    if (!csvFile.is_open()) {
      throw std::runtime_error("Could not open file: " + pathToFile);
    }

    for (size_t i{}; i < m_points.n; ++i) {
      for (size_t dim{}; dim < ndim; ++dim) {
        csvFile << m_points.m_coords << ',';
      }
      csvFile << m_points.m_clusterIndex[i] << ',' << m_points.m_isSeed[i];
      csvFile << '\n';
    }
  }
};

#endif
