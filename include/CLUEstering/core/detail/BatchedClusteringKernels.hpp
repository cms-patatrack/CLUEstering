
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/default_allocator.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/nostd/ceil_div.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <alpaka/alpaka.hpp>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>

CLUE_NAMESPACE_BEGIN
namespace detail {

  struct KernelCalculateLocalDensityBatched {
    template <typename TAcc,
              std::size_t Ndim,
              std::floating_point TData,
              concepts::convolutional_kernel KernelType,
              concepts::distance_metric<Ndim> DistanceMetric,
              std::floating_point TPointsData = TData>
      requires(alpaka::Dim<TAcc>::value == 2 &&
               std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TPointsData> dev_points,
                                  const KernelType& kernel,
                                  TData density_radius,
                                  DistanceMetric metric,
                                  const auto* event_offsets,
                                  std::size_t max_event_size,
                                  std::size_t /* blocks_per_event */) const {
      for (auto event : alpaka::uniformElementsAlong<0u>(acc)) {
        for (auto local_idx : alpaka::uniformElementsAlong<1u>(acc, max_event_size)) {
          const auto global_idx = event_offsets[event] + local_idx;
          if (global_idx < event_offsets[event + 1]) {
            auto rho_i = TData{0};
            auto coords_i = dev_points[global_idx];

            clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
            for (auto dim = 0u; dim != Ndim; ++dim) {
              const auto sigma_i =
                  dev_points.has_sigma(dim) ? dev_points.sigma(dim)[global_idx] : TData{0};
              const auto box_radius =
                  math::max(density_radius, density_radius * sigma_i * math::sqrt(TData{2}));
              searchbox_extremes[dim] =
                  clue::nostd::make_array(coords_i[dim] - box_radius, coords_i[dim] + box_radius);
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
                                            global_idx,
                                            event);

            assert(rho_i >= TData{0});
            dev_points.rho()[global_idx] = rho_i;
          }
        }
      }
    }
  };

  struct KernelCalculateNearestHigherBatched {
    template <typename TAcc,
              std::size_t Ndim,
              std::floating_point TData,
              concepts::distance_metric<Ndim> DistanceMetric,
              std::floating_point TPointsData = TData>
      requires(alpaka::Dim<TAcc>::value == 2 &&
               std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TPointsData> dev_points,
                                  TData outlier_distance,
                                  TData seeding_distance,
                                  TData min_density,
                                  DistanceMetric metric,
                                  std::size_t* nclusters,
                                  const auto* event_offsets,
                                  std::size_t max_event_size,
                                  std::size_t /* blocks_per_event */) const {
      for (auto event : alpaka::uniformElementsAlong<0u>(acc)) {
        for (auto local_idx : alpaka::uniformElementsAlong<1u>(acc, max_event_size)) {
          const auto global_idx = event_offsets[event] + local_idx;
          if (global_idx < event_offsets[event + 1]) {
            auto delta_i = std::numeric_limits<TData>::max();
            int nh_i = -1;
            auto coords_i = dev_points[global_idx];
            auto rho_i = dev_points.rho()[global_idx];
            const auto density_uncertainty = dev_points.has_uncertainty()
                                                 ? dev_points.density_uncertainty()[global_idx]
                                                 : TData{1.};
            const auto effective_min_density = min_density * density_uncertainty;
            const auto effective_distance =
                (rho_i >= effective_min_density) ? seeding_distance : outlier_distance;

            clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
            for (auto dim = 0u; dim != Ndim; ++dim) {
              const auto sigma_i =
                  dev_points.has_sigma(dim) ? dev_points.sigma(dim)[global_idx] : TData{0};
              const auto box_radius = math::max(
                  effective_distance, effective_distance * sigma_i * math::sqrt(TData{2}));
              searchbox_extremes[dim] =
                  clue::nostd::make_array(coords_i[dim] - box_radius, coords_i[dim] + box_radius);
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
                                                           effective_distance,
                                                           metric,
                                                           global_idx,
                                                           event);

            assert(nh_i == -1 || delta_i <= effective_distance);
            dev_points.nearest_higher()[global_idx] = nh_i;
            if (dev_points.has_delta()) {
              dev_points.delta()[global_idx] =
                  (nh_i == -1) ? std::numeric_limits<TData>::infinity() : delta_i;
            }

            const auto is_seed = (nh_i == -1) && (rho_i >= effective_min_density);
            if (is_seed) {
              dev_points.is_seed()[global_idx] = 1;
              // guarantees seed indices from 0 to Nseeds-1 across the whole batch
              const auto slot = alpaka::atomicAdd(acc, nclusters, std::size_t{1});
              dev_points.cluster_index()[global_idx] = static_cast<int>(slot);
            } else {
              dev_points.is_seed()[global_idx] = 0;
              dev_points.cluster_index()[global_idx] = -1;
            }
          }
        }
      }
    }
  };

  struct KernelEventOfSeeds {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  PointsView<Ndim, TData> dev_points,
                                  const std::size_t* event_offsets,
                                  std::size_t batch_size,
                                  std::int32_t* cluster_to_event) const {
      for (auto i : alpaka::uniformElements(acc, dev_points.size())) {
        if (dev_points.is_seed()[i] != 0) {
          // binary search of the event e such that event_offsets[e] <= i < event_offsets[e+1]
          std::size_t lo = 0;
          std::size_t hi = batch_size;
          while (hi - lo > 1) {
            const auto mid = lo + (hi - lo) / 2;
            if (event_offsets[mid] <= i) {
              lo = mid;
            } else {
              hi = mid;
            }
          }
          cluster_to_event[dev_points.cluster_index()[i]] = static_cast<std::int32_t>(lo);
        }
      }
    }
  };

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData>
    requires(alpaka::Dim<TAcc>::value == 2 &&
             std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
  inline void computeLocalDensityBatched(TQueue& queue,
                                         internal::TilesView<Ndim, TData>& tiles,
                                         PointsView<Ndim, TPointsData>& dev_points,
                                         KernelType&& kernel,
                                         TData density_radius,
                                         const DistanceMetric& metric,
                                         const auto& event_offsets,
                                         std::size_t max_event_size,
                                         std::size_t block_size) {
    const auto blocks_per_event = nostd::ceil_div(max_event_size, block_size);
    const auto batch_size = alpaka::getExtents(event_offsets)[0] - 1;
    const auto work_division =
        make_workdiv<internal::Acc2D>({batch_size, blocks_per_event}, {1, block_size});
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelCalculateLocalDensityBatched{},
                       tiles,
                       dev_points,
                       std::forward<KernelType>(kernel),
                       density_radius,
                       metric,
                       event_offsets.data(),
                       max_event_size,
                       blocks_per_event);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData>
    requires(alpaka::Dim<TAcc>::value == 2 &&
             std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
  inline void computeNearestHighersBatched(TQueue& queue,
                                           internal::TilesView<Ndim, TData>& tiles,
                                           PointsView<Ndim, TPointsData>& dev_points,
                                           TData outlier_distance,
                                           TData seeding_distance,
                                           TData min_density,
                                           const DistanceMetric& metric,
                                           std::size_t* d_nclusters,
                                           const auto& event_offsets,
                                           std::size_t max_event_size,
                                           std::size_t block_size) {
    const auto blocks_per_event = nostd::ceil_div(max_event_size, block_size);
    const auto batch_size = alpaka::getExtents(event_offsets)[0] - 1;
    const auto work_division =
        make_workdiv<internal::Acc2D>({batch_size, blocks_per_event}, {1, block_size});
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelCalculateNearestHigherBatched{},
                       tiles,
                       dev_points,
                       outlier_distance,
                       seeding_distance,
                       min_density,
                       metric,
                       d_nclusters,
                       event_offsets.data(),
                       max_event_size,
                       blocks_per_event);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData>
    requires(alpaka::Dim<TAcc>::value == 1)
  inline void eventOfSeeds(TQueue& queue,
                           std::size_t block_size,
                           PointsView<Ndim, TData> dev_points,
                           const std::size_t* event_offsets,
                           std::size_t batch_size,
                           std::int32_t* cluster_to_event) {
    const Idx grid_size = nostd::ceil_div(dev_points.size(), block_size);
    alpaka::exec<TAcc>(queue,
                       clue::make_workdiv<TAcc>(grid_size, block_size),
                       KernelEventOfSeeds{},
                       dev_points,
                       event_offsets,
                       batch_size,
                       cluster_to_event);
  }

}  // namespace detail
CLUE_NAMESPACE_END
