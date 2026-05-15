
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/internal/DeviceVector.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/nostd/ceil_div.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <alpaka/alpaka.hpp>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace clue::detail {

  template <typename TAcc,
            std::size_t Ndim,
            std::size_t N_,
            std::floating_point TData,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric>
  ALPAKA_FN_ACC void for_recursion(const TAcc& acc,
                                   std::array<int32_t, Ndim>& base_vec,
                                   const clue::SearchBoxBins<Ndim>& search_box,
                                   internal::TilesView<Ndim, TData>& tiles,
                                   PointsView<Ndim, TData>& dev_points,
                                   const KernelType& kernel,
                                   const std::array<TData, Ndim + 1>& coords_i,
                                   TData& rho_i,
                                   TData density_radius,
                                   const DistanceMetric& metric,
                                   int32_t point_id,
                                   std::size_t event = 0) {
    if constexpr (N_ == 0) {
      auto tile_idx = tiles.getGlobalBinByBin(base_vec, event);
      auto tile_size = tiles[tile_idx].size();

      for (auto tile_it = 0u; tile_it < tile_size; ++tile_it) {
        auto j = tiles[tile_idx][tile_it];
        assert(j >= 0 && j < dev_points.size());

        auto coords_j = dev_points[j];
        auto distance = metric(coords_i, coords_j);
        assert(distance >= TData{0});

        auto k = kernel(acc, distance, point_id, j);
        assert(k >= TData{0});
        rho_i += static_cast<int>(distance <= density_radius) * k * dev_points.weights()[j];
      }
      return;
    } else {
      for (auto i = search_box[search_box.size() - N_][0];
           i <= search_box[search_box.size() - N_][1];
           ++i) {
        base_vec[Ndim - N_] = i;
        for_recursion<TAcc, Ndim, N_ - 1>(acc,
                                          base_vec,
                                          search_box,
                                          tiles,
                                          dev_points,
                                          kernel,
                                          coords_i,
                                          rho_i,
                                          density_radius,
                                          metric,
                                          point_id,
                                          event);
      }
    }
  }

  struct KernelCalculateLocalDensity {
    template <typename TAcc,
              std::size_t Ndim,
              std::floating_point TData,
              concepts::convolutional_kernel KernelType,
              concepts::distance_metric<Ndim> DistanceMetric>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TData> dev_points,
                                  const KernelType& kernel,
                                  TData density_radius,
                                  DistanceMetric metric,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        auto rho_i = static_cast<TData>(0.);
        auto coords_i = dev_points[i];

        clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          searchbox_extremes[dim] = clue::nostd::make_array(coords_i[dim] - density_radius,
                                                            coords_i[dim] + density_radius);
        }

        clue::SearchBoxBins<Ndim> searchbox_bins;
        dev_tiles.searchBox(searchbox_extremes, searchbox_bins);

        std::array<int32_t, Ndim> base_vec;
        for_recursion<TAcc, Ndim, Ndim>(acc,
                                        base_vec,
                                        searchbox_bins,
                                        dev_tiles,
                                        dev_points,
                                        kernel,
                                        coords_i,
                                        rho_i,
                                        density_radius,
                                        metric,
                                        i);

        assert(rho_i >= TData{0});
        dev_points.rho()[i] = rho_i;
      }
    }
  };

  template <typename TAcc,
            std::size_t Ndim,
            std::size_t N_,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric>
  ALPAKA_FN_ACC void for_recursion_nearest_higher(const TAcc& acc,
                                                  std::array<int32_t, Ndim>& base_vec,
                                                  const clue::SearchBoxBins<Ndim>& search_box,
                                                  internal::TilesView<Ndim, TData>& tiles,
                                                  PointsView<Ndim, TData>& dev_points,
                                                  const std::array<TData, Ndim + 1>& coords_i,
                                                  TData rho_i,
                                                  TData& delta_i,
                                                  int& nh_i,
                                                  TData outlier_distance,
                                                  TData seeding_distance,
                                                  const DistanceMetric& metric,
                                                  int32_t point_id,
                                                  std::size_t event = 0) {
    if constexpr (N_ == 0) {
      auto tile_idx = tiles.getGlobalBinByBin(base_vec, event);
      auto tile_size = tiles[tile_idx].size();

      for (auto tile_it = 0u; tile_it < tile_size; ++tile_it) {
        const auto j = tiles[tile_idx][tile_it];
        assert(j >= 0 && j < dev_points.size());
        auto rho_j = dev_points.rho()[j];
        bool found_higher_in_tile = (rho_j > rho_i);
        found_higher_in_tile =
            found_higher_in_tile || ((rho_j == rho_i) && (rho_j > TData{0}) && (j > point_id));

        if (found_higher_in_tile) {
          auto coords_j = dev_points[j];
          auto distance = metric(coords_i, coords_j);
          assert(distance >= TData{0});

          if (distance <= outlier_distance && distance < delta_i) {
            delta_i = distance;
            nh_i = (distance > seeding_distance) ? -1 : j;
          }
        }
      }

      return;
    } else {
      for (auto i = search_box[search_box.size() - N_][0];
           i <= search_box[search_box.size() - N_][1];
           ++i) {
        base_vec[Ndim - N_] = i;
        for_recursion_nearest_higher<TAcc, Ndim, N_ - 1>(acc,
                                                         base_vec,
                                                         search_box,
                                                         tiles,
                                                         dev_points,
                                                         coords_i,
                                                         rho_i,
                                                         delta_i,
                                                         nh_i,
                                                         outlier_distance,
                                                         seeding_distance,
                                                         metric,
                                                         point_id,
                                                         event);
      }
    }
  }

  struct KernelCalculateNearestHigher {
    template <typename TAcc,
              std::size_t Ndim,
              std::floating_point TData,
              concepts::distance_metric<Ndim> DistanceMetric>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TData> dev_points,
                                  TData outlier_distance,
                                  TData seeding_distance,
                                  DistanceMetric metric,
                                  std::size_t* seed_candidates,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        auto delta_i = std::numeric_limits<TData>::max();
        int nh_i = -1;
        auto coords_i = dev_points[i];
        auto rho_i = dev_points.rho()[i];

        clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          searchbox_extremes[dim] = clue::nostd::make_array(coords_i[dim] - outlier_distance,
                                                            coords_i[dim] + outlier_distance);
        }

        clue::SearchBoxBins<Ndim> searchbox_bins;
        dev_tiles.searchBox(searchbox_extremes, searchbox_bins);

        std::array<int32_t, Ndim> base_vec{};
        for_recursion_nearest_higher<TAcc, Ndim, Ndim>(acc,
                                                       base_vec,
                                                       searchbox_bins,
                                                       dev_tiles,
                                                       dev_points,
                                                       coords_i,
                                                       rho_i,
                                                       delta_i,
                                                       nh_i,
                                                       outlier_distance,
                                                       seeding_distance,
                                                       metric,
                                                       i);

        assert(nh_i == -1 || delta_i <= outlier_distance);
        dev_points.nearest_higher()[i] = nh_i;
        if (nh_i == -1) {
          alpaka::atomicAdd(acc, seed_candidates, std::size_t{1});
        }
      }
    }
  };

  struct KernelFindClusters {
    template <typename TAcc,
              std::size_t Ndim,
              std::floating_point TData,
              concepts::distance_metric<Ndim> DistanceMetric>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> dev_points,
                                  TData seeding_distance,
                                  DistanceMetric metric,
                                  TData min_density,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        dev_points.cluster_index()[i] = -1;
        auto nh = dev_points.nearest_higher()[i];

        auto coords_i = dev_points[i];
        auto coords_nh = dev_points[nh];
        auto distance = metric(coords_i, coords_nh);
        assert(distance >= TData{0});

        auto rho_i = dev_points.rho()[i];
        const auto density_uncertainty =
            (dev_points.has_uncertainty()) ? dev_points.density_uncertainty()[i] : TData{1.};
        bool is_seed =
            (distance > seeding_distance) && (rho_i >= min_density * density_uncertainty);

        if (is_seed) {
          dev_points.is_seed()[i] = 1;
          dev_points.nearest_higher()[i] = -1;
          seeds.push_back(acc, i);
        } else {
          dev_points.is_seed()[i] = 0;
        }
      }
    }
  };

  struct KernelAssignSeedIndices {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> dev_points) const {
      for (auto cls_idx : alpaka::uniformElements(acc, seeds.size())) {
        dev_points.cluster_index()[seeds[cls_idx]] = static_cast<int>(cls_idx);
      }
    }
  };

  struct KernelAssignClusters {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  PointsView<Ndim, TData> dev_points,
                                  int32_t n_points) const {
      for (auto idx : alpaka::uniformElements(acc, n_points)) {
        if (dev_points.is_seed()[idx] || dev_points.nearest_higher()[idx] == -1)
          continue;

        auto current = idx;
        while (!dev_points.is_seed()[current] && dev_points.nearest_higher()[current] != -1)
          current = dev_points.nearest_higher()[current];

        dev_points.cluster_index()[idx] = dev_points.cluster_index()[current];
      }
    }
  };

  using WorkDiv = clue::WorkDiv<clue::Dim1D>;

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric>
  inline void computeLocalDensity(TQueue& queue,
                                  const WorkDiv& work_division,
                                  internal::TilesView<Ndim, TData>& tiles,
                                  PointsView<Ndim, TData>& dev_points,
                                  KernelType&& kernel,
                                  TData density_radius,
                                  const DistanceMetric& metric,
                                  int32_t size) {
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelCalculateLocalDensity{},
                       tiles,
                       dev_points,
                       std::forward<KernelType>(kernel),
                       density_radius,
                       metric,
                       size);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric>
    requires(alpaka::Dim<TAcc>::value == 1)
  inline void computeNearestHighers(TQueue& queue,
                                    const WorkDiv& work_division,
                                    internal::TilesView<Ndim, TData>& tiles,
                                    PointsView<Ndim, TData>& dev_points,
                                    TData outlier_distance,
                                    TData seeding_distance,
                                    const DistanceMetric& metric,
                                    std::size_t& seed_candidates,
                                    int32_t size) {
    auto d_seed_candidates = clue::make_device_buffer<std::size_t>(queue);
    alpaka::memset(queue, d_seed_candidates, 0u);
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelCalculateNearestHigher{},
                       tiles,
                       dev_points,
                       outlier_distance,
                       seeding_distance,
                       metric,
                       d_seed_candidates.data(),
                       size);
    alpaka::memcpy(queue, clue::make_host_view(seed_candidates), d_seed_candidates);
    alpaka::wait(queue);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric>
    requires(alpaka::Dim<TAcc>::value == 1)
  inline void findClusterSeeds(TQueue& queue,
                               const WorkDiv& work_division,
                               clue::internal::SeedArray<>& seeds,
                               PointsView<Ndim, TData>& dev_points,
                               TData seeding_distance,
                               const DistanceMetric& metric,
                               TData min_density,
                               int32_t size) {
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelFindClusters{},
                       seeds.view(),
                       dev_points,
                       seeding_distance,
                       metric,
                       min_density,
                       size);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData>
  inline void assignPointsToClusters(TQueue& queue,
                                     std::size_t block_size,
                                     clue::internal::SeedArray<>& seeds,
                                     PointsView<Ndim, TData> dev_points,
                                     int32_t n_points) {
    const Idx seed_grid = nostd::ceil_div(seeds.size(queue), block_size);
    alpaka::exec<TAcc>(queue,
                       clue::make_workdiv<TAcc>(seed_grid, block_size),
                       KernelAssignSeedIndices{},
                       seeds.view(),
                       dev_points);

    const Idx point_grid = nostd::ceil_div(n_points, block_size);
    alpaka::exec<TAcc>(queue,
                       clue::make_workdiv<TAcc>(point_grid, block_size),
                       KernelAssignClusters{},
                       dev_points,
                       n_points);
  }

}  // namespace clue::detail
