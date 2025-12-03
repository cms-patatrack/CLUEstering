
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/detail/get_cluster_properties.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test batched clustering with fixed batch size") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/batched_data_1024.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  const std::size_t batch_size = 1024;

  std::vector<uint32_t> event_sizes(10, batch_size);
  algo.make_clusters(queue, h_points, d_points, event_sizes);
  alpaka::wait(queue);

  auto truth = clue::read_output<2>(queue, "../../../data/truth_files/data_1024_truth.csv");
  auto truth_n_clusters = clue::detail::compute_nclusters(truth.clusterIndexes());
  auto n_clusters = clue::detail::compute_nclusters(h_points.clusterIndexes());
  CHECK(n_clusters == truth_n_clusters * 10);
}
