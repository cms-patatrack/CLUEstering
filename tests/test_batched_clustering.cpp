
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/detail/get_cluster_properties.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("Test batched clustering with fixed batch size") {
  SUBCASE("Test from host points") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    clue::PointsHost<2> h_points =
        clue::read_csv<2, float>(queue, "../../../data/batched_data_1024.csv");
    const auto n_points = h_points.size();
    clue::PointsDevice<2> d_points(queue, n_points);

    const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
    const std::size_t batch_size = 1024;

    std::vector<uint32_t> event_sizes(10, batch_size);
    algo.make_clusters(queue, h_points, d_points, event_sizes);
    alpaka::wait(queue);

    auto truth =
        clue::read_output<2, float>(queue, "../../../data/truth_files/data_1024_truth.csv");
    auto truth_n_clusters = clue::detail::compute_nclusters(truth.clusterIndexes());
    auto n_clusters = clue::detail::compute_nclusters(h_points.clusterIndexes());
    CHECK(n_clusters == truth_n_clusters * 10);

    auto sample_cluster_associations = algo.getSampleAssociations(queue, h_points);
    CHECK(sample_cluster_associations.size() == 10);
  }
  SUBCASE("Test from device points") {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    clue::PointsHost<2> h_points =
        clue::read_csv<2, float>(queue, "../../../data/batched_data_1024.csv");
    const auto n_points = h_points.size();

    clue::PointsDevice<2> d_points(queue, n_points);
    clue::copyToDevice(queue, d_points, h_points);
    alpaka::wait(queue);

    const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
    clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
    const std::size_t batch_size = 1024;

    std::vector<uint32_t> event_sizes(10, batch_size);
    algo.make_clusters(queue, d_points, event_sizes);

    clue::copyToHost(queue, h_points, d_points);
    alpaka::wait(queue);

    auto truth =
        clue::read_output<2, float>(queue, "../../../data/truth_files/data_1024_truth.csv");
    auto truth_n_clusters = clue::detail::compute_nclusters(truth.clusterIndexes());
    auto n_clusters = clue::detail::compute_nclusters(h_points.clusterIndexes());
    CHECK(n_clusters == truth_n_clusters * 10);

    auto sample_cluster_associations = algo.getSampleAssociations(queue, d_points);
    CHECK(sample_cluster_associations.size() == 10);
  }
}
