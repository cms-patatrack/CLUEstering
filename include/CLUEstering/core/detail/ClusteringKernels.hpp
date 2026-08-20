
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
#include <limits>
#include <type_traits>

namespace clue::detail {

  template <typename TAcc,
            std::size_t Ndim,
            std::size_t N_,
            std::floating_point TData,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData>
    requires std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>
  ALPAKA_FN_ACC void for_recursion(const TAcc& acc,
                                   std::array<int32_t, Ndim>& base_vec,
                                   const clue::SearchBoxBins<Ndim>& search_box,
                                   internal::TilesView<Ndim, TData>& tiles,
                                   PointsView<Ndim, TPointsData>& points,
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
        assert(j >= 0 && j < points.size());

        const auto distance = [&]() -> TData {
          if constexpr (concepts::detail::view_distance_metric<DistanceMetric, Ndim>) {
            return metric(points, static_cast<std::size_t>(point_id), static_cast<std::size_t>(j));
          } else {
            return metric(coords_i, points[j]);
          }
        }();
        assert(distance >= TData{0});

        auto k = kernel(distance, point_id, j);
        assert(k >= TData{0});
        rho_i += static_cast<int>(distance <= density_radius) * k * points.weights()[j];
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
                                          points,
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
              concepts::distance_metric<Ndim> DistanceMetric,
              std::floating_point TPointsData = TData>
      requires(alpaka::Dim<TAcc>::value == 1 &&
               std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> tiles,
                                  PointsView<Ndim, TPointsData> points,
                                  const KernelType& kernel,
                                  TData density_radius,
                                  DistanceMetric metric) const {
      for (auto i : alpaka::uniformElements(acc, points.size())) {
        auto rho_i = static_cast<TData>(0.);
        auto coords_i = points[i];

        clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          const auto sigma_i = points.has_sigma(dim) ? points.sigma(dim)[i] : TData{0};
          const auto box_radius =
              math::max(density_radius, density_radius * sigma_i * math::sqrt(TData{2}));
          searchbox_extremes[dim] =
              clue::nostd::make_array(coords_i[dim] - box_radius, coords_i[dim] + box_radius);
        }

        clue::SearchBoxBins<Ndim> searchbox_bins;
        tiles.searchBox(searchbox_extremes, searchbox_bins);

        std::array<int32_t, Ndim> base_vec;
        for_recursion<TAcc, Ndim, Ndim>(acc,
                                        base_vec,
                                        searchbox_bins,
                                        tiles,
                                        points,
                                        kernel,
                                        coords_i,
                                        rho_i,
                                        density_radius,
                                        metric,
                                        i);

        assert(rho_i >= TData{0});
        points.rho()[i] = rho_i;
      }
    }
  };

  template <typename TAcc,
            std::size_t Ndim,
            std::size_t N_,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData>
    requires std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>
  ALPAKA_FN_ACC void for_recursion_nearest_higher(const TAcc& acc,
                                                  std::array<int32_t, Ndim>& base_vec,
                                                  const clue::SearchBoxBins<Ndim>& search_box,
                                                  internal::TilesView<Ndim, TData>& tiles,
                                                  PointsView<Ndim, TPointsData>& points,
                                                  const std::array<TData, Ndim + 1>& coords_i,
                                                  TData rho_i,
                                                  TData& delta_i,
                                                  int& nh_i,
                                                  TData outlier_distance,
                                                  TData seeding_distance,
                                                  TData min_density,
                                                  const DistanceMetric& metric,
                                                  int32_t point_id,
                                                  std::size_t event = 0) {
    if constexpr (N_ == 0) {
      auto tile_idx = tiles.getGlobalBinByBin(base_vec, event);
      auto tile_size = tiles[tile_idx].size();

      const auto effective_distance = (rho_i >= min_density) ? seeding_distance : outlier_distance;

      auto tag = [&points](std::integral auto idx) -> std::size_t {
        return (points.has_tags()) ? points.tags()[idx] : static_cast<std::size_t>(idx);
      };

      auto point_tag = tag(point_id);
      for (auto tile_it = 0u; tile_it < tile_size; ++tile_it) {
        const auto j = tiles[tile_idx][tile_it];
        const auto tag_j = tag(j);
        assert(j >= 0 && j < points.size());
        auto rho_j = points.rho()[j];
        bool found_higher_in_tile = (rho_j > rho_i);
        found_higher_in_tile =
            found_higher_in_tile || ((rho_j == rho_i) && (rho_j > TData{0}) && (tag_j > point_tag));

        if (found_higher_in_tile) {
          const auto distance = [&]() -> TData {
            if constexpr (concepts::detail::view_distance_metric<DistanceMetric, Ndim>) {
              return metric(
                  points, static_cast<std::size_t>(point_id), static_cast<std::size_t>(j));
            } else {
              return metric(coords_i, points[j]);
            }
          }();
          assert(distance >= TData{0});

          if (distance <= effective_distance &&
              ((distance < delta_i) ||
               ((distance == delta_i) && (nh_i >= 0) &&
                ((rho_j > points.rho()[nh_i]) ||
                 ((rho_j == points.rho()[nh_i]) && (tag_j > tag(nh_i))))))) {
            delta_i = distance;
            nh_i = j;
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
                                                         points,
                                                         coords_i,
                                                         rho_i,
                                                         delta_i,
                                                         nh_i,
                                                         outlier_distance,
                                                         seeding_distance,
                                                         min_density,
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
              concepts::distance_metric<Ndim> DistanceMetric,
              std::floating_point TPointsData = TData>
      requires(alpaka::Dim<TAcc>::value == 1 &&
               std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> tiles,
                                  PointsView<Ndim, TPointsData> points,
                                  TData outlier_distance,
                                  TData seeding_distance,
                                  TData min_density,
                                  DistanceMetric metric,
                                  std::size_t* seed_candidates) const {
      for (auto i : alpaka::uniformElements(acc, points.size())) {
        auto delta_i = std::numeric_limits<TData>::max();
        int nh_i = -1;
        auto coords_i = points[i];
        auto rho_i = points.rho()[i];
        const auto density_uncertainty =
            points.has_uncertainty() ? points.density_uncertainty()[i] : TData{1.};
        const auto effective_min_density = min_density * density_uncertainty;

        clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          const auto sigma_i = points.has_sigma(dim) ? points.sigma(dim)[i] : TData{0};
          const auto box_radius =
              math::max(outlier_distance, outlier_distance * sigma_i * math::sqrt(TData{2}));
          searchbox_extremes[dim] =
              clue::nostd::make_array(coords_i[dim] - box_radius, coords_i[dim] + box_radius);
        }

        clue::SearchBoxBins<Ndim> searchbox_bins;
        tiles.searchBox(searchbox_extremes, searchbox_bins);

        std::array<int32_t, Ndim> base_vec{};
        for_recursion_nearest_higher<TAcc, Ndim, Ndim>(acc,
                                                       base_vec,
                                                       searchbox_bins,
                                                       tiles,
                                                       points,
                                                       coords_i,
                                                       rho_i,
                                                       delta_i,
                                                       nh_i,
                                                       outlier_distance,
                                                       seeding_distance,
                                                       effective_min_density,
                                                       metric,
                                                       i);

        assert(nh_i == -1 || delta_i <= outlier_distance);
        points.nearest_higher()[i] = nh_i;
        if (nh_i == -1) {
          alpaka::atomicAdd(acc, seed_candidates, std::size_t{1});
        }
      }
    }
  };

  struct KernelFindClusters {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> points,
                                  TData min_density) const {
      for (auto i : alpaka::uniformElements(acc, points.size())) {
        points.cluster_index()[i] = -1;
        const auto nh = points.nearest_higher()[i];
        const auto rho_i = points.rho()[i];
        const auto density_uncertainty =
            points.has_uncertainty() ? points.density_uncertainty()[i] : TData{1.};
        const auto is_seed = (nh == -1) && (rho_i >= min_density * density_uncertainty);

        if (is_seed) {
          points.is_seed()[i] = 1;
          seeds.push_back(acc, i);
        } else {
          points.is_seed()[i] = 0;
        }
      }
    }
  };

  struct KernelAssignSeedIndices {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> points) const {
      for (auto cls_idx : alpaka::uniformElements(acc, seeds.size())) {
        points.cluster_index()[seeds[cls_idx]] = static_cast<int>(cls_idx);
      }
    }
  };

  struct KernelAssignClusters {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, PointsView<Ndim, TData> points) const {
      for (auto idx : alpaka::uniformElements(acc, points.size())) {
        if (points.is_seed()[idx] || points.nearest_higher()[idx] == -1)
          continue;

        auto current = idx;
        while (!points.is_seed()[current] && points.nearest_higher()[current] != -1)
          current = points.nearest_higher()[current];

        points.cluster_index()[idx] = points.cluster_index()[current];
      }
    }
  };

  using WorkDiv = clue::WorkDiv<clue::Dim1D>;

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData>
    requires std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>
  inline void computeLocalDensity(TQueue& queue,
                                  const WorkDiv& work_division,
                                  internal::TilesView<Ndim, TData>& tiles,
                                  PointsView<Ndim, TPointsData>& points,
                                  KernelType&& kernel,
                                  TData density_radius,
                                  const DistanceMetric& metric) {
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelCalculateLocalDensity{},
                       tiles,
                       points,
                       std::forward<KernelType>(kernel),
                       density_radius,
                       metric);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData>
    requires(alpaka::Dim<TAcc>::value == 1 &&
             std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
  inline void computeNearestHighers(TQueue& queue,
                                    const WorkDiv& work_division,
                                    internal::TilesView<Ndim, TData>& tiles,
                                    PointsView<Ndim, TPointsData>& points,
                                    TData outlier_distance,
                                    TData seeding_distance,
                                    TData min_density,
                                    const DistanceMetric& metric,
                                    std::size_t& seed_candidates) {
    auto d_seed_candidates = clue::make_device_buffer<std::size_t>(queue);
    alpaka::memset(queue, d_seed_candidates, 0u);
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelCalculateNearestHigher{},
                       tiles,
                       points,
                       outlier_distance,
                       seeding_distance,
                       min_density,
                       metric,
                       d_seed_candidates.data());
    alpaka::memcpy(queue, clue::make_host_view(seed_candidates), d_seed_candidates);
    alpaka::wait(queue);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData>
    requires(alpaka::Dim<TAcc>::value == 1)
  inline void findClusterSeeds(TQueue& queue,
                               const WorkDiv& work_division,
                               clue::internal::SeedArray<>& seeds,
                               PointsView<Ndim, TData>& points,
                               TData min_density) {
    alpaka::exec<TAcc>(
        queue, work_division, KernelFindClusters{}, seeds.view(), points, min_density);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData>
  inline void assignPointsToClusters(TQueue& queue,
                                     std::size_t block_size,
                                     clue::internal::SeedArray<>& seeds,
                                     PointsView<Ndim, TData> points) {
    const auto nseeds = seeds.size(queue);
    if (nseeds == 0) {
      alpaka::fill(queue,
                   clue::make_device_view(
                       alpaka::getDev(queue), points.cluster_index().data(), points.size()),
                   -1);
      return;
    }

    const Idx seed_grid = nostd::ceil_div(nseeds, block_size);
    alpaka::exec<TAcc>(queue,
                       clue::make_workdiv<TAcc>(seed_grid, block_size),
                       KernelAssignSeedIndices{},
                       seeds.view(),
                       points);

    const Idx point_grid = nostd::ceil_div(points.size(), block_size);
    alpaka::exec<TAcc>(
        queue, clue::make_workdiv<TAcc>(point_grid, block_size), KernelAssignClusters{}, points);
  }

}  // namespace clue::detail
