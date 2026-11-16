
#include "CLUEstering/CLUEstering.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

namespace {

  struct Scenario {
    clue::Queue queue;
    clue::PointsHost<2> h_points;
    clue::PointsDevice<2> d_points;
    clue::Clusterer<2> algo;
    std::vector<uint32_t> event_sizes;

    Scenario(bool batched, bool store_seeds = true)
        : queue(clue::get_device(0u)),
          h_points(clue::read_csv<2, float>(queue,
                                            batched
                                                ? std::string("../../../data/batched_data_1024.csv")
                                                : std::string(TEST_DATA_DIR) + "/data_1024.csv")),
          d_points(queue, h_points.size()),
          algo(batched ? clue::Clusterer<2>(1.3f, 10.f, 1.3f)
                       : clue::Clusterer<2>(1.5f, 10.f, 1.5f)) {
      if (store_seeds) {
        h_points.store_seeds();
      }
      if (batched) {
        event_sizes = std::vector<uint32_t>(10, 1024);
        algo.make_clusters(queue, h_points, d_points, event_sizes);
      } else {
        algo.make_clusters(queue, h_points, d_points);
      }
      alpaka::wait(queue);
    }
  };

  std::vector<int32_t> device_seeds(Scenario& s) {
    auto d_seeds = s.algo.getSeeds(s.queue, s.d_points);
    std::vector<int32_t> seeds(static_cast<std::size_t>(s.d_points.n_clusters()));
    if (!seeds.empty()) {
      alpaka::memcpy(s.queue, alpaka::createView(clue::host, seeds.data(), seeds.size()), d_seeds);
    }
    alpaka::wait(s.queue);
    return seeds;
  }

  void check_seeds(const std::vector<int32_t>& seeds, const clue::PointsHost<2>& h_points) {
    const auto cluster_index = h_points.clusterIndexes();
    std::set<int32_t> unique;
    for (std::size_t c = 0; c < seeds.size(); ++c) {
      REQUIRE(seeds[c] >= 0);
      REQUIRE(static_cast<std::size_t>(seeds[c]) < cluster_index.size());
      CHECK(cluster_index[seeds[c]] == static_cast<int>(c));
      unique.insert(seeds[c]);
    }
    CHECK(unique.size() == seeds.size());
  }

  constexpr std::size_t n_events = 10;
  constexpr std::size_t event_size = 1024;

  // copy an association map to the host, returning for each key the list of its values
  template <typename TMap>
  std::vector<std::vector<int32_t>> map_to_groups(clue::Queue& queue, const TMap& map) {
    const auto nkeys = static_cast<std::size_t>(map.size());
    const auto nvalues = static_cast<std::size_t>(map.extents().values);
    std::vector<int32_t> offsets(nkeys + 1), indexes(nvalues);
    const auto containers = map.extract();
    alpaka::memcpy(
        queue, alpaka::createView(clue::host, offsets.data(), offsets.size()), containers.keys);
    if (nvalues > 0) {
      alpaka::memcpy(
          queue, alpaka::createView(clue::host, indexes.data(), indexes.size()), containers.values);
    }
    alpaka::wait(queue);
    std::vector<std::vector<int32_t>> groups(nkeys);
    for (std::size_t k = 0; k < nkeys; ++k) {
      groups[k].assign(indexes.begin() + offsets[k], indexes.begin() + offsets[k + 1]);
    }
    return groups;
  }

  // every cluster must appear under exactly one event, which must be the one of its seed
  void check_associations(const std::vector<std::vector<int32_t>>& groups,
                          const std::vector<int32_t>& seeds,
                          std::size_t n_clusters) {
    CHECK(groups.size() == n_events);
    std::map<int32_t, std::size_t> event_of_cluster;
    std::size_t total = 0;
    for (std::size_t e = 0; e < groups.size(); ++e) {
      total += groups[e].size();
      for (const auto c : groups[e]) {
        CHECK(event_of_cluster.emplace(c, e).second);
      }
    }
    CHECK(total == n_clusters);
    CHECK(event_of_cluster.size() == n_clusters);
    REQUIRE(seeds.size() == n_clusters);
    for (std::size_t c = 0; c < n_clusters; ++c) {
      const auto it = event_of_cluster.find(static_cast<int32_t>(c));
      REQUIRE(it != event_of_cluster.end());
      CHECK(it->second == static_cast<std::size_t>(seeds[c]) / event_size);
    }
  }

}  // namespace

TEST_CASE("Test the device getSeeds overload") {
  SUBCASE("Throws on points that were never clustered") {
    clue::Queue queue(clue::get_device(0u));
    clue::Clusterer<2> algo(1.5f, 10.f, 1.5f);
    clue::PointsDevice<2> d_points(queue, 1024);
    CHECK_THROWS_AS(algo.getSeeds(queue, d_points), std::runtime_error);
  }
  SUBCASE("Non-batched") {
    Scenario s(false);
    const auto seeds = device_seeds(s);
    CHECK(seeds.size() == static_cast<std::size_t>(s.d_points.n_clusters()));
    CHECK(seeds.size() > 0);
    check_seeds(seeds, s.h_points);
  }
  SUBCASE("Batched") {
    Scenario s(true);
    const auto seeds = device_seeds(s);
    CHECK(seeds.size() == static_cast<std::size_t>(s.d_points.n_clusters()));
    CHECK(seeds.size() > 0);
    check_seeds(seeds, s.h_points);
  }
}

TEST_CASE("Test the host getSeeds overload") {
  SUBCASE("Throws on points that were never clustered") {
    clue::Queue queue(clue::get_device(0u));
    clue::PointsHost<2> h_points(queue, 1024);
    clue::Clusterer<2> algo(1.5f, 10.f, 1.5f);
    CHECK_THROWS_AS(algo.getSeeds(h_points), std::runtime_error);
  }
  SUBCASE("Throws on points that never went through make_clusters") {
    clue::Queue queue(clue::get_device(0u));
    clue::Clusterer<2> algo(1.5f, 10.f, 1.5f);
    auto h_points = clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/data_1024.csv");
    CHECK_THROWS_AS(algo.getSeeds(h_points), std::runtime_error);
  }
  for (const bool batched : {false, true}) {
    SUBCASE(batched ? "Batched" : "Non-batched") {
      Scenario s(batched);
      const auto seeds = s.algo.getSeeds(s.h_points);
      CHECK(seeds.size() == static_cast<std::size_t>(s.h_points.n_clusters()));
      CHECK(seeds.size() > 0);
      check_seeds(seeds, s.h_points);
      CHECK(seeds == device_seeds(s));
    }
  }
}

TEST_CASE("Test the opt-in storage of the seeds on the host") {
  SUBCASE("Not stored by default, with host and device points") {
    for (const bool batched : {false, true}) {
      Scenario s(batched, false);
      CHECK_FALSE(s.h_points.has_seeds());
      CHECK_THROWS_AS(s.algo.getSeeds(s.h_points), std::runtime_error);
      CHECK_THROWS_AS(s.h_points.seed_flags(), std::runtime_error);
    }
  }
  SUBCASE("Not stored by default, with the host points only") {
    clue::Queue queue(clue::get_device(0u));
    auto h_points = clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/data_1024.csv");
    clue::Clusterer<2> algo(1.5f, 10.f, 1.5f);
    algo.make_clusters(queue, h_points);
    alpaka::wait(queue);
    CHECK_FALSE(h_points.has_seeds());
    CHECK_THROWS_AS(algo.getSeeds(h_points), std::runtime_error);
  }
  SUBCASE("store_seeds is idempotent") {
    clue::Queue queue(clue::get_device(0u));
    clue::PointsHost<2> h_points(queue, 1024);
    CHECK_FALSE(h_points.has_seeds());
    h_points.store_seeds();
    CHECK(h_points.has_seeds());
    const auto* data = h_points.seed_flags().data();
    h_points.store_seeds();
    CHECK(h_points.has_seeds());
    CHECK(h_points.seed_flags().data() == data);
    CHECK(h_points.seed_flags().size() == 1024);
  }
  SUBCASE("Host-only make_clusters, non-batched") {
    clue::Queue queue(clue::get_device(0u));
    auto h_points = clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/data_1024.csv");
    h_points.store_seeds();
    clue::Clusterer<2> algo(1.5f, 10.f, 1.5f);
    algo.make_clusters(queue, h_points);
    alpaka::wait(queue);
    const auto seeds = algo.getSeeds(h_points);
    CHECK(seeds.size() == static_cast<std::size_t>(h_points.n_clusters()));
    CHECK(seeds.size() > 0);
    check_seeds(seeds, h_points);
  }
  SUBCASE("Host-only make_clusters, batched") {
    clue::Queue queue(clue::get_device(0u));
    auto h_points =
        clue::read_csv<2, float>(queue, std::string("../../../data/batched_data_1024.csv"));
    h_points.store_seeds();
    clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);
    const std::vector<uint32_t> sizes(n_events, event_size);
    algo.make_clusters(queue, h_points, std::span<const uint32_t>(sizes));
    alpaka::wait(queue);
    const auto seeds = algo.getSeeds(h_points);
    CHECK(seeds.size() == static_cast<std::size_t>(h_points.n_clusters()));
    CHECK(seeds.size() > 0);
    check_seeds(seeds, h_points);
  }
}

TEST_CASE("Test the cluster ids of the batched clustering") {
  Scenario s(true);
  const auto cluster_index = s.h_points.clusterIndexes();
  const auto n_clusters = static_cast<std::size_t>(clue::detail::compute_nclusters(cluster_index));
  CHECK(n_clusters == static_cast<std::size_t>(s.d_points.n_clusters()));

  // every cluster id belongs to exactly one event
  std::map<int32_t, std::set<std::size_t>> events_of_cluster;
  for (std::size_t i = 0; i < cluster_index.size(); ++i) {
    if (cluster_index[i] >= 0) {
      events_of_cluster[cluster_index[i]].insert(i / event_size);
    }
  }
  for (const auto& [cluster, events] : events_of_cluster) {
    CHECK(events.size() == 1);
  }

  // the ids used are exactly 0..N-1
  CHECK(events_of_cluster.size() == n_clusters);
  std::size_t expected = 0;
  for (const auto& [cluster, events] : events_of_cluster) {
    CHECK(cluster == static_cast<int32_t>(expected++));
  }

  // each event has the same number of clusters as the truth
  auto truth =
      clue::read_output<2, float>(s.queue, "../../../data/truth_files/data_1024_truth.csv");
  const auto truth_n_clusters =
      static_cast<std::size_t>(clue::detail::compute_nclusters(truth.clusterIndexes()));
  std::vector<std::size_t> clusters_per_event(n_events, 0);
  for (const auto& [cluster, events] : events_of_cluster) {
    ++clusters_per_event[*events.begin()];
  }
  for (const auto n : clusters_per_event) {
    CHECK(n == truth_n_clusters);
  }
}

TEST_CASE("Test getSampleAssociations") {
  SUBCASE("Device overload") {
    Scenario s(true);
    const auto seeds = device_seeds(s);
    auto map = s.algo.getSampleAssociations(s.queue, s.d_points);
    CHECK(map.size() == n_events);
    check_associations(map_to_groups(s.queue, map), seeds, seeds.size());
  }
  SUBCASE("Host overload") {
    Scenario s(true);
    const auto seeds = s.algo.getSeeds(s.h_points);
    auto map = s.algo.getSampleAssociations(s.queue, s.h_points);
    CHECK(map.size() == n_events);
    check_associations(map_to_groups(s.queue, map), seeds, seeds.size());
  }
  SUBCASE("Throws on points that were never clustered") {
    clue::Queue queue(clue::get_device(0u));
    clue::Clusterer<2> algo(1.3f, 10.f, 1.3f);
    clue::PointsDevice<2> d_points(queue, 1024);
    clue::PointsHost<2> h_points(queue, 1024);
    CHECK_THROWS_AS(algo.getSampleAssociations(queue, d_points), std::runtime_error);
    CHECK_THROWS_AS(algo.getSampleAssociations(queue, h_points), std::runtime_error);
  }
  SUBCASE("Throws if no batched clustering was run") {
    Scenario s(false);
    CHECK_THROWS_AS(s.algo.getSampleAssociations(s.queue, s.d_points), std::runtime_error);
    CHECK_THROWS_AS(s.algo.getSampleAssociations(s.queue, s.h_points), std::runtime_error);
  }
}
