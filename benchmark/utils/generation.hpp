
#include <algorithm>
#include <execution>
#include <random>
#include <vector>

#include "CLUEstering/data_structures/PointsHost.hpp"

namespace clue {
  namespace utils {

    template <uint8_t Ndim>
    using ClusterCenters = std::vector<std::array<float, Ndim>>;

    namespace detail {

      template <uint8_t Ndim>
      ClusterCenters<Ndim> computeClusterCenters(size_t n_clusters,
                                                 std::uniform_real_distribution<float> uniform_dist,
                                                 std::mt19937& gen) {
        ClusterCenters<Ndim> cluster_centers(n_clusters);

        std::generate(cluster_centers.begin(), cluster_centers.end(), [&]() {
          std::array<float, Ndim> center;
          std::generate(std::execution::unseq, center.begin(), center.end(), [&]() {
            return uniform_dist(gen);
          });
          return center;
        });

        return cluster_centers;
      }

      template <uint8_t Ndim>
      void generateClusterData(clue::PointsHost<Ndim>& points,
                               const ClusterCenters<Ndim>& cluster_centers,
                               size_t n_clusters,
                               size_t cluster_size,
                               float stddev,
                               std::mt19937& gen) {
        for (auto cluster = 0u; cluster < n_clusters; ++cluster) {
          for (auto dim = 0u; dim < Ndim; ++dim) {
            std::normal_distribution<float> gaussian_dist(cluster_centers[cluster][dim], stddev);
            std::generate_n(std::execution::unseq,
                            points.coords(dim).begin() + cluster * cluster_size,
                            cluster_size,
                            [&]() { return gaussian_dist(gen); });
          }
        }
      }

      template <uint8_t Ndim>
      void generateNoiseData(clue::PointsHost<Ndim>& points,
                             size_t data_size,
                             size_t noise_size,
                             std::uniform_real_distribution<float> uniform_dist,
                             std::mt19937& gen) {
        for (auto dim = 0u; dim < Ndim; ++dim) {
          std::generate_n(
              std::execution::unseq, points.coords(dim).begin() + data_size, noise_size, [&]() {
                return uniform_dist(gen);
              });
        }
      }

    }  // namespace detail

    template <uint8_t Ndim>
    void generateRandomData(clue::PointsHost<Ndim>& points,
                            size_t n_clusters,
                            std::pair<float, float> space_boundaries,
                            float stddev,
                            float noisiness = .1f,
                            int seed = 0) {
      std::mt19937 gen(seed);
      std::uniform_real_distribution<float> uniform_dist(space_boundaries.first,
                                                         space_boundaries.second);

      auto cluster_centers = detail::computeClusterCenters<Ndim>(n_clusters, uniform_dist, gen);

      const auto cluster_size =
          static_cast<size_t>(std::floor(points.size() * (1 - noisiness) / n_clusters));
      const auto data_size = n_clusters * cluster_size;
      const auto noise_size = points.size() - data_size;
      detail::generateClusterData<Ndim>(
          points, cluster_centers, n_clusters, cluster_size, stddev, gen);
      detail::generateNoiseData<Ndim>(points, data_size, noise_size, uniform_dist, gen);
      std::fill(points.weights().begin(), points.weights().end(), 1.0f);
    }

  }  // namespace utils
}  // namespace clue
