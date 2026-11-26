#include "CLUEstering/CLUEstering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

namespace {

  struct Parameters {
    float density_radius;
    float min_density;
  };

  struct Result {
    std::vector<int32_t> labels;
    std::vector<float> rho;
    std::vector<int32_t> nearest_higher;
  };

  // Cluster the points of a dataset in the data directory and return the cluster indexes
  Result run(const std::string& dataset, Parameters params, std::optional<float> merge_threshold) {
    const auto device = clue::get_device(0u);
    clue::Queue queue(device);

    auto h_points = clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/" + dataset);
    h_points.store_details();
    clue::Clusterer<2> algo(params.density_radius, params.min_density);
    algo.setMergeThreshold(merge_threshold);
    algo.make_clusters(queue, h_points);

    const auto labels = h_points.clusterIndexes();
    const auto rho = h_points.rho();
    const auto nearest_higher = h_points.nearest_higher();
    return Result{{labels.begin(), labels.end()},
                  {rho.begin(), rho.end()},
                  {nearest_higher.begin(), nearest_higher.end()}};
  }

  // Number of clusters among the points from first to last - 1, ignoring the outliers
  std::size_t count_clusters(const std::vector<int32_t>& labels,
                             std::size_t first = 0,
                             std::size_t last = std::size_t(-1)) {
    last = std::min(last, labels.size());
    std::set<int32_t> clusters;
    for (auto i = first; i < last; ++i) {
      if (labels[i] >= 0)
        clusters.insert(labels[i]);
    }
    return clusters.size();
  }

  // Whether the merged clusters are unions of the plain ones, with the same outliers
  bool is_coarsening(const std::vector<int32_t>& plain, const std::vector<int32_t>& merged) {
    if (plain.size() != merged.size())
      return false;
    std::map<int32_t, int32_t> merged_of;
    for (auto i = 0u; i < plain.size(); ++i) {
      if ((plain[i] < 0) != (merged[i] < 0))
        return false;
      if (plain[i] < 0)
        continue;
      const auto [it, inserted] = merged_of.try_emplace(plain[i], merged[i]);
      if (!inserted && it->second != merged[i])
        return false;
    }
    return true;
  }

  // The cluster indexes go from 0 to the number of clusters - 1, and in each merged cluster the
  // points without a nearest higher (the seeds of the plain clusters) have at most the density
  // of a point of the cluster, so the merged cluster keeps the highest peak
  void check_labels(const Result& result) {
    const auto n_clusters = count_clusters(result.labels);
    CHECK(std::ranges::max(result.labels) + 1 == static_cast<int32_t>(n_clusters));

    std::vector<float> peak(n_clusters, 0.f);
    std::vector<float> max_rho(n_clusters, 0.f);
    for (auto i = 0u; i < result.labels.size(); ++i) {
      const auto label = result.labels[i];
      if (label < 0)
        continue;
      max_rho[label] = std::max(max_rho[label], result.rho[i]);
      if (result.nearest_higher[i] == -1)
        peak[label] = std::max(peak[label], result.rho[i]);
    }
    CHECK(peak == max_rho);
  }

  constexpr Parameters moons_params{30.f, 8.f};
  constexpr Parameters blob_pairs_params{.5f, 8.f};
  constexpr Parameters noise_params{.02f, 1.f};

  // In blob_pairs.csv the points of the k-th pair of blobs are the rows from 1000 * k
  constexpr std::size_t pair_size = 1000;
  constexpr std::size_t n_pairs = 3;

  std::vector<std::size_t> clusters_per_pair(const std::vector<int32_t>& labels) {
    std::vector<std::size_t> counts;
    for (auto k = 0u; k < n_pairs; ++k)
      counts.push_back(count_clusters(labels, k * pair_size, (k + 1) * pair_size));
    return counts;
  }

}  // namespace

TEST_CASE("Test the validation of the merge threshold") {
  clue::Clusterer<2> algo(1.f, 1.f);
  CHECK_NOTHROW(algo.setMergeThreshold(std::nullopt));
  CHECK_NOTHROW(algo.setMergeThreshold(1.f));
  CHECK_NOTHROW(algo.setMergeThreshold(.01f));
  CHECK_THROWS_AS(algo.setMergeThreshold(0.f), std::invalid_argument);
  CHECK_THROWS_AS(algo.setMergeThreshold(-.5f), std::invalid_argument);
  CHECK_THROWS_AS(algo.setMergeThreshold(1.5f), std::invalid_argument);
}

TEST_CASE("Test that the merging is not supported for batched clustering") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  auto h_points = clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/moons.csv");
  const auto n_points = static_cast<uint32_t>(h_points.size());
  clue::Clusterer<2> algo(moons_params.density_radius, moons_params.min_density);
  algo.setMergeThreshold(.5f);
  std::vector<uint32_t> batch_sizes{n_points / 2, n_points - n_points / 2};
  CHECK_THROWS_AS(algo.make_clusters(queue, h_points, batch_sizes), std::logic_error);
}

TEST_CASE("Test that the merging disabled gives the plain CLUE results") {
  const auto plain = run("moons.csv", moons_params, std::nullopt);

  const auto device = clue::get_device(0u);
  clue::Queue queue(device);
  auto h_points = clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/moons.csv");
  h_points.store_details();
  clue::Clusterer<2> algo(moons_params.density_radius, moons_params.min_density);
  // enabling and then disabling the merging gives back plain CLUE
  algo.setMergeThreshold(.5f);
  algo.setMergeThreshold(std::nullopt);
  algo.make_clusters(queue, h_points);

  // the same partition, the cluster indexes can be permuted by the parallel backends
  const auto labels = h_points.clusterIndexes();
  const std::vector<int32_t> disabled(labels.begin(), labels.end());
  CHECK(is_coarsening(plain.labels, disabled));
  CHECK(count_clusters(disabled) == count_clusters(plain.labels));
}

TEST_CASE("Test the merging on the moons dataset") {
  // plain CLUE splits each moon in several clusters, the merging gives back the two moons
  const auto plain = run("moons.csv", moons_params, std::nullopt);
  CHECK(count_clusters(plain.labels) > 2);

  // the counts are collected and checked together, so that a failure shows the threshold
  std::vector<std::size_t> n_clusters;
  for (const auto threshold : {.8f, .5f, .1f}) {
    const auto merged = run("moons.csv", moons_params, threshold);
    n_clusters.push_back(count_clusters(merged.labels));
    CHECK(is_coarsening(plain.labels, merged.labels));
    check_labels(merged);
  }
  CHECK(n_clusters == std::vector<std::size_t>{2, 2, 2});
}

TEST_CASE("Test the merging of pairs of blobs with valleys of different depth") {
  // The blobs of each pair are at 2.5, 4 and 8 sigma. The first pair merges for thresholds up
  // to about 0.85, the second up to about 0.25, the third never, because its blobs do not touch.
  const auto plain = run("blob_pairs.csv", blob_pairs_params, std::nullopt);
  for (const auto n : clusters_per_pair(plain.labels))
    CHECK(n >= 2);

  const auto shallow = run("blob_pairs.csv", blob_pairs_params, .6f);
  CHECK(clusters_per_pair(shallow.labels) == std::vector<std::size_t>{1, 2, 2});
  CHECK(is_coarsening(plain.labels, shallow.labels));
  check_labels(shallow);

  const auto deep = run("blob_pairs.csv", blob_pairs_params, .1f);
  CHECK(clusters_per_pair(deep.labels) == std::vector<std::size_t>{1, 1, 2});
  CHECK(is_coarsening(plain.labels, deep.labels));
  CHECK(is_coarsening(shallow.labels, deep.labels));
  check_labels(deep);

  // no threshold merges the blobs that do not touch
  const auto lowest = run("blob_pairs.csv", blob_pairs_params, .01f);
  CHECK(clusters_per_pair(lowest.labels)[2] == 2);
}

TEST_CASE("Test the merging of many touching clusters") {
  // Hundreds of small clusters touching each other. The cluster pairs are more than the initial
  // capacity of the buffer of the saddles, so the saddles are computed again in a larger buffer.
  const auto plain = run("uniform_noise.csv", noise_params, std::nullopt);
  const auto n_plain = count_clusters(plain.labels);
  CHECK(n_plain > 100);

  const auto partial = run("uniform_noise.csv", noise_params, .9f);
  const auto n_partial = count_clusters(partial.labels);
  CHECK(n_partial < n_plain);
  CHECK(n_partial > 1);
  CHECK(is_coarsening(plain.labels, partial.labels));
  check_labels(partial);

  const auto full = run("uniform_noise.csv", noise_params, .05f);
  CHECK(count_clusters(full.labels) == 1);
  CHECK(is_coarsening(plain.labels, full.labels));
  CHECK(is_coarsening(partial.labels, full.labels));
  check_labels(full);
}
