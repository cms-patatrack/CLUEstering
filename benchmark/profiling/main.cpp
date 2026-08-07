#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"

void run_one(clue::Queue& queue, const std::string& input_file, const std::string& tag, FILE* summary) {
  auto h_points = clue::read_csv<2, float>(queue, input_file);
  clue::PointsDevice<2> d_points(queue, h_points.size());

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

 // clue::GaussianKernel<float, 2> kernel{0.f, 1.5f, 1.f};
  clue::FlatKernel<float, 2> kernel{0.5f, dc};   

  const auto t0 = std::chrono::steady_clock::now();
  algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<2, float>{}, kernel);
  const auto t1 = std::chrono::steady_clock::now();

  auto clusters = algo.getClusters(h_points);

  auto n_negative = 0u;
  auto min_rho = d_points.view().rho()[0];
  for (auto i = 0u; i < h_points.size(); ++i) {
    const auto r = d_points.view().rho()[i];
    if (r < 0.f) ++n_negative;
    if (r < min_rho) min_rho = r;
  }

  const auto name = std::filesystem::path(input_file).stem().string();
  const auto out_csv = tag + "_" + name + ".csv";
  FILE* f = std::fopen(out_csv.c_str(), "w");
  std::fprintf(f, "x,y,rho\n");
  for (auto i = 0u; i < h_points.size(); ++i) {
    std::fprintf(f, "%.9f,%.9f,%.9f\n",
                 d_points.view()[i][0], d_points.view()[i][1], d_points.view().rho()[i]);
  }
  std::fclose(f);

  std::fprintf(summary, "%s,%zu,%zu,%f,%u,%f\n",
               name.c_str(), h_points.size(), clusters.size(),
               std::chrono::duration<double, std::milli>(t1 - t0).count(),
               n_negative, min_rho);
  std::printf("done: %s -> %s\n", name.c_str(), out_csv.c_str());
}

int main(int argc, char* argv[]) {
  auto tag{std::string(argv[1])};        
  auto data_dir{std::string(argv[2])};   

  std::vector<std::string> datasets;
  for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
    if (entry.path().extension() == ".csv") {
      datasets.push_back(entry.path().string());
    }
  }
  std::sort(datasets.begin(), datasets.end());

  std::printf("found %zu datasets in %s\n", datasets.size(), data_dir.c_str());

  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto summary_file = tag + "_summary.csv";
  FILE* summary = std::fopen(summary_file.c_str(), "w");
  std::fprintf(summary, "dataset,n_points,n_clusters,time_ms,n_negative,min_rho\n");

  for (const auto& ds : datasets) {
    run_one(queue, ds, tag, summary);
  }
  std::fclose(summary);
}