
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
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
                                  std::size_t* seed_candidates,
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
                                                           effective_min_density,
                                                           metric,
                                                           global_idx,
                                                           event);

            assert(nh_i == -1 || delta_i <= outlier_distance);
            dev_points.nearest_higher()[global_idx] = nh_i;
            if (nh_i == -1) {
              alpaka::atomicAdd(acc, seed_candidates, std::size_t{1});
            }
          }
        }
      }
    }
  };

  struct KernelFindClustersBatched {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
      requires(alpaka::Dim<TAcc>::value == 2)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> dev_points,
                                  std::remove_cv_t<TData> min_density,
                                  clue::internal::DeviceVectorView event_associations,
                                  const auto* event_offsets,
                                  std::size_t max_event_size) const {
      for (const auto event : alpaka::uniformElementsAlong<0u>(acc)) {
        for (const auto local_idx : alpaka::uniformElementsAlong<1u>(acc, max_event_size)) {
          const auto global_idx = event_offsets[event] + local_idx;
          if (global_idx < event_offsets[event + 1]) {
            dev_points.cluster_index()[global_idx] = -1;
            const auto nh = dev_points.nearest_higher()[global_idx];
            const auto rho_i = dev_points.rho()[global_idx];
            const auto density_uncertainty = dev_points.has_uncertainty()
                                                 ? dev_points.density_uncertainty()[global_idx]
                                                 : TData{1.};
            const auto is_seed = (nh == -1) && (rho_i >= min_density * density_uncertainty);

            if (is_seed) {
              dev_points.is_seed()[global_idx] = 1;
              const auto prev = seeds.push_back(acc, global_idx);
              event_associations[prev] = event;
            } else {
              dev_points.is_seed()[global_idx] = 0;
            }
          }
        }
      }
    }
  };

  struct KernelReorderSeeds {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView old_seeds,
                                  clue::internal::SeedArrayView old_batches,
                                  const int32_t* batches_to_seeds_indexes,
                                  clue::internal::SeedArrayView new_seeds,
                                  clue::internal::SeedArrayView new_batches,
                                  std::size_t num_seeds) const {
      for (auto ii : alpaka::uniformElements(acc, num_seeds)) {
        auto old_seed_idx = batches_to_seeds_indexes[ii];
        new_seeds[ii] = old_seeds[old_seed_idx];
        new_batches[ii] = old_batches[old_seed_idx];
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
                                           std::size_t& seed_candidates,
                                           const auto& event_offsets,
                                           std::size_t max_event_size,
                                           std::size_t block_size) {
    auto d_seed_candidates = clue::make_device_buffer<std::size_t>(queue);
    alpaka::memset(queue, d_seed_candidates, 0u);

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
                       d_seed_candidates.data(),
                       event_offsets.data(),
                       max_event_size,
                       blocks_per_event);
    alpaka::memcpy(queue, clue::make_host_view(seed_candidates), d_seed_candidates);
    alpaka::wait(queue);
  }

  template <concepts::accelerator TAcc, concepts::queue TQueue>
    requires(alpaka::Dim<TAcc>::value == 1)
  inline void reorderSeedsBatchWise(TQueue& queue,
                                    clue::internal::SeedArray<>& seeds,
                                    clue::internal::SeedArray<>& batch_association) {
    const auto num_seeds = seeds.size(queue);
    if (num_seeds == 0)
      return;

    auto batches_to_seeds = clue::internal::make_associator(
        queue,
        std::span<const int32_t>{batch_association.data(), num_seeds},
        static_cast<int32_t>(num_seeds));

    auto extracted = batches_to_seeds.extract();
    auto* batches_to_seeds_indexes = extracted.values.data();

    auto seeds_reordered = clue::internal::SeedArray<>(queue, num_seeds);
    auto batches_reordered = clue::internal::SeedArray<>(queue, num_seeds);

    const auto block_size = 512;
    const auto grid_size = clue::divide_up_by(num_seeds, block_size);
    const auto work_division = clue::make_workdiv<internal::Acc>(grid_size, block_size);

    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelReorderSeeds{},
                       seeds.view(),
                       batch_association.view(),
                       batches_to_seeds_indexes,
                       seeds_reordered.view(),
                       batches_reordered.view(),
                       num_seeds);

    alpaka::memcpy(
        queue,
        clue::make_device_view(alpaka::getDev(queue), seeds.data(), num_seeds),
        clue::make_device_view(alpaka::getDev(queue), seeds_reordered.data(), num_seeds));

    alpaka::memcpy(
        queue,
        clue::make_device_view(alpaka::getDev(queue), batch_association.data(), num_seeds),
        clue::make_device_view(alpaka::getDev(queue), batches_reordered.data(), num_seeds));

    alpaka::wait(queue);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData>
    requires(alpaka::Dim<TAcc>::value == 2)
  inline void findClusterSeedsBatched(TQueue& queue,
                                      clue::internal::SeedArray<>& seeds,
                                      PointsView<Ndim, TData>& dev_points,
                                      std::remove_cv_t<TData> min_density,
                                      const auto& event_offsets,
                                      std::size_t max_event_size,
                                      const clue::internal::DeviceVectorView& event_associations,
                                      std::size_t block_size) {
    const auto blocks_per_event = nostd::ceil_div(max_event_size, block_size);
    const auto batch_size = alpaka::getExtents(event_offsets)[0] - 1;
    const auto work_division =
        make_workdiv<internal::Acc2D>({batch_size, blocks_per_event}, {1, block_size});
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelFindClustersBatched{},
                       seeds.view(),
                       dev_points,
                       min_density,
                       event_associations,
                       event_offsets.data(),
                       max_event_size);
  }

}  // namespace clue::detail
