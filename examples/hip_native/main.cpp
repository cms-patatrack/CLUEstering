
#include <CLUEstering/CLUEstering.hpp>
#include <hip_runtime.h>
#include <vector>

template <typename TQueue>
void compute_clusters(TQueue& queue, std::vector<int>& cluster_indexes) {
  // Allocate the points on the host and device.
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, "../../data/sissa_1000.csv");
  clue::PointsDevice<2> d_points(queue, h_points.size());

  // Define the parameters for the clustering and construct the clusterer.
  const float dc = 20.f, rhoc = 10.f, outlier = 20.f;
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  // Launch the clustering
  // The results will be stored in the `clue::PointsHost` object
  algo.make_clusters(queue, h_points, d_points);

  std::ranges::copy(h_points.clusterIndexes(), std::back_inserter(cluster_indexes));
}

int main() {
  hipStream_t stream;
  hipStreamCreate(&stream);
  clue::Queue queue(stream);

  std::vector<int> cluster_indexes;
  compute_clusters(queue, cluster_indexes);

  hipStreamDestroy(stream);
}
