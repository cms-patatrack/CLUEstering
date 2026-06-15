
#include "CLUEstering/CLUEstering.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("Test to_dot export") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_1024.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2> d_points(queue, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);

  const std::string dot_file_path = "test_clusters.dot";
  clue::to_dot(queue, d_points, dot_file_path);

  REQUIRE(std::filesystem::exists(dot_file_path));

  std::ifstream file(dot_file_path);
  REQUIRE(file.is_open());
  std::ostringstream ss;
  ss << file.rdbuf();
  const std::string content = ss.str();

  CHECK(!content.empty());
  CHECK(content.substr(0, 12) == "digraph G {\n");
  CHECK(content.substr(content.size() - 2) == "}\n");

  // Each of the n_points nodes appears exactly once as a definition line.
  // Node lines are indented with 4 spaces followed by a digit (the node index).
  int node_count = 0;
  std::istringstream line_stream(content);
  std::string line;
  while (std::getline(line_stream, line)) {
    if (line.size() >= 5 && line[0] == ' ' && line[1] == ' ' && line[2] == ' ' && line[3] == ' ' &&
        std::isdigit(static_cast<unsigned char>(line[4]))) {
      ++node_count;
    }
  }
  CHECK(node_count == static_cast<int>(n_points));
}

TEST_CASE("Test to_dot throws on unclustered points") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_1024.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
  clue::PointsDevice<2> d_points(queue, h_points.size());

  const auto call = [&] { clue::to_dot(queue, d_points, "should_not_exist.dot"); };
  CHECK_THROWS_AS(call(), std::runtime_error);
}
