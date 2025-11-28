
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test batched clustering with fixed batch size") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  clue::PointsHost<2> h_points = clue::read_csv<2>(queue, "../../../data/batched_data.csv");
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  const std::size_t batch_size = 8192;

  algo.make_clusters(queue, h_points, d_points, clue::FlatKernel{.5f}, batch_size);

  const auto batches = n_points / batch_size;
  for (auto batch = 0u; batches < batches; ++batch) {
    clue::PointsHost<2> batch_points(queue,
                                     batch_size,
                                     h_points.coords(0).data() + batch * batch_size,
                                     h_points.coords(1).data() + batch * batch_size,
                                     h_points.weights().data() + batch * batch_size,
                                     h_points.clusterIndexes().data() + batch * batch_size);
    auto truth = clue::read_output<2>(queue, "../../../data/truth_files/data_8192.csv");
    CHECK(clue::validate_results(batch_points, truth));
  }
}
