
#include <CLUEstering/CLUEstering.hpp>

int main() {
  // Obtain the queue, which is used for allocations and kernel launches.
  auto queue = clue::get_queue(0u);

  // Allocate the points on the host and device.
  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../data/sissa.csv");
  clue::PointsDevice<2> d_points(queue, h_points.size());

  // Define the parameters for the clustering and construct the clusterer.
  const float dc = 20.f, rhoc = 10.f, outlier = 20.f;
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  // Launch the clustering
  // The results will be stored in the `clue::PointsHost` object
  algo.make_clusters(queue, h_points, d_points, clue::FlatKernel{.5f});
  // Read the data from the host points
  auto clusters_indexes = h_points.clusterIndexes();  // Get the cluster index for each points
  auto seed_map =
      h_points.isSeed();  // Obtain a boolean array indicating which points are the seeds
                          // i.e. the cluster centers
}
