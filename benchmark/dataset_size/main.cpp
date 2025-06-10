
#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/DataFormats/PointsHost.hpp"
#include "CLUEstering/DataFormats/PointsDevice.hpp"

#include "utils/generation.hpp"

#ifdef PYBIND11
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
#endif

struct TimeMeasures {
  std::vector<int> sizes;
  std::vector<float> time_averages;
  std::vector<float> time_stddevs;

  TimeMeasures(size_t size) : sizes(size), time_averages(size), time_stddevs(size) {}
};

float mean(const std::vector<float>& values) {
  return std::accumulate(values.begin(), values.end(), 0.f) / values.size();
}

float stddev(const std::vector<float>& values) {
  auto mean_ = mean(values);
  auto view = values |
              std::views::transform([mean_](auto x) -> float { return (x - mean_) * (x - mean_); });
  auto sqSize = values.size() * (values.size() - 1);
  return std::sqrt(std::accumulate(view.begin(), view.end(), 0.f) / sqSize);
}

#ifdef PYBIND11
void plot(const TimeMeasures& measures, const std::string& filename) {
  py::scoped_interpreter guard{};
  py::module plt = py::module::import("matplotlib.pyplot");
  py::bind_vector<std::vector<int>>(plt, "VectorInt");
  py::bind_vector<std::vector<float>>(plt, "VectorFloat");
  plt.attr("errorbar")(
      measures.sizes, measures.time_averages, measures.time_stddevs, "fmt"_a = "r--^");
  plt.attr("xlabel")("Number of points");
  plt.attr("ylabel")("Execution time (ms)");
  plt.attr("grid")("ls"_a = "--", "lw"_a = .5);
  plt.attr("savefig")(filename);
}
#endif

void to_csv(const TimeMeasures& measures, const std::string& filename) {
  std::ofstream file{filename};
  if (!file.is_open()) {
    std::cerr << "Error opening file " << filename << std::endl;
    return;
  }

  file << "size,avg,std\n";
  for (auto i = 0ul; i < measures.sizes.size(); ++i) {
    file << measures.sizes[i] << "," << measures.time_averages[i] << "," << measures.time_stddevs[i]
         << "\n";
  }
  file.close();
}

using ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Acc1D;
using ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Device;
using ALPAKA_ACCELERATOR_NAMESPACE_CLUE::Queue;

void run(clue::PointsHost<2>& h_points, clue::PointsDevice<2, Device>& d_points, Queue& queue) {
  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<2> algo(queue, dc, rhoc, outlier);

  const std::size_t block_size{256};
  algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
}

int main(int argc, char* argv[]) {
  auto min = std::stoi(argv[1]);
  auto max = std::stoi(argv[2]);
  auto range = max - min + 1;

  std::string oFilename{"measures.csv"};
  if (argc == 4) {
    oFilename = argv[3];
  }

  TimeMeasures measures(range);
  auto& sizes = measures.sizes;
  auto& time_averages = measures.time_averages;
  auto& time_stddevs = measures.time_stddevs;
  auto nruns{10};

  auto avgIt = time_averages.begin();
  auto stdIt = time_stddevs.begin();
  auto sizeIt = sizes.begin();
  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  std::ranges::for_each(
      std::views::iota(min) | std::views::take(range),
      [nruns, &device, &sizeIt, &avgIt, &stdIt](auto i) -> void {
        Queue queue(device);

        const auto n_points = static_cast<std::size_t>(std::pow(2, i));

        // Create the points host and device objects
        clue::PointsHost<2> h_points(queue, n_points);
        clue::PointsDevice<2, Device> d_points(queue, n_points);
        clue::utils::generateRandomData<2>(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::vector<float> times(nruns);
        for (auto i = 0; i < nruns; ++i) {
          start = std::chrono::high_resolution_clock::now();
          run(h_points, d_points, queue);
          end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
          times[i] = duration;
        }
        *sizeIt++ = n_points;
        *avgIt++ = mean(times);
        *stdIt++ = stddev(times);
      });

#ifdef PYBIND11
  auto figname = oFilename.substr(0, oFilename.find_last_of('.')) + ".pdf";
  plot(measures, figname);
#endif
  to_csv(measures, oFilename);
}
