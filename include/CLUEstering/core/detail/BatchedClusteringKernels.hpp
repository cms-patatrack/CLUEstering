
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/internal/DeviceVector.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/data_structures/internal/VecArray.hpp"
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
              concepts::distance_metric<Ndim> DistanceMetric>
      requires(alpaka::Dim<TAcc>::value == 2)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TData> dev_points,
                                  const KernelType& kernel,
                                  TData dc,
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
              searchbox_extremes[dim] =
                  clue::nostd::make_array(coords_i[dim] - dc, coords_i[dim] + dc);
            }

            clue::SearchBoxBins<Ndim> searchbox_bins;
            dev_tiles.searchBox(searchbox_extremes, searchbox_bins);

            VecArray<int32_t, Ndim> base_vec;
            for_recursion<TAcc, Ndim, Ndim>(acc,
                                            base_vec,
                                            searchbox_bins,
                                            dev_tiles,
                                            dev_points,
                                            kernel,
                                            coords_i,
                                            rho_i,
                                            dc,
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
              concepts::distance_metric<Ndim> DistanceMetric>
      requires(alpaka::Dim<TAcc>::value == 2)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TData> dev_points,
                                  TData dm,
                                  TData seed_dc,
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

            clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
            for (auto dim = 0u; dim != Ndim; ++dim) {
              searchbox_extremes[dim] =
                  clue::nostd::make_array(coords_i[dim] - dm, coords_i[dim] + dm);
            }

            clue::SearchBoxBins<Ndim> searchbox_bins;
            dev_tiles.searchBox(searchbox_extremes, searchbox_bins);

            VecArray<int32_t, Ndim> base_vec{};
            for_recursion_nearest_higher<TAcc, Ndim, Ndim>(acc,
                                                           base_vec,
                                                           searchbox_bins,
                                                           dev_tiles,
                                                           dev_points,
                                                           coords_i,
                                                           rho_i,
                                                           delta_i,
                                                           nh_i,
                                                           dm,
                                                           seed_dc,
                                                           metric,
                                                           global_idx,
                                                           event);

            assert(nh_i == -1 || delta_i <= dm);
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
    template <typename TAcc,
              std::size_t Ndim,
              std::floating_point TData,
              concepts::distance_metric<Ndim> DistanceMetric>
      requires(alpaka::Dim<TAcc>::value == 2)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> dev_points,
                                  TData seed_dc,
                                  DistanceMetric metric,
                                  TData rhoc,
                                  clue::internal::DeviceVectorView event_associations,
                                  const auto* event_offsets,
                                  std::size_t max_event_size) const {
      for (const auto event : alpaka::uniformElementsAlong<0u>(acc)) {
        for (const auto local_idx : alpaka::uniformElementsAlong<1u>(acc, max_event_size)) {
          const auto global_idx = event_offsets[event] + local_idx;
          if (global_idx < event_offsets[event + 1]) {
            dev_points.cluster_index()[global_idx] = -1;
            auto nh = dev_points.nearest_higher()[global_idx];

            auto coords_i = dev_points[global_idx];
            auto coords_nh = dev_points[nh];
            auto distance = metric(coords_i, coords_nh);
            assert(distance >= TData{0});

            auto rho_i = dev_points.rho()[global_idx];
            bool is_seed = (distance > seed_dc) && (rho_i >= rhoc);

            if (is_seed) {
              dev_points.is_seed()[global_idx] = 1;
              dev_points.nearest_higher()[global_idx] = -1;
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

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric>
    requires(alpaka::Dim<TAcc>::value == 2)
  inline void computeLocalDensityBatched(TQueue& queue,
                                         internal::TilesView<Ndim, TData>& tiles,
                                         PointsView<Ndim, TData>& dev_points,
                                         KernelType&& kernel,
                                         TData dc,
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
                       dc,
                       metric,
                       event_offsets.data(),
                       max_event_size,
                       blocks_per_event);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric>
    requires(alpaka::Dim<TAcc>::value == 2)
  inline void computeNearestHighersBatched(TQueue& queue,
                                           internal::TilesView<Ndim, TData>& tiles,
                                           PointsView<Ndim, TData>& dev_points,
                                           TData dm,
                                           TData seed_dc,
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
                       dm,
                       seed_dc,
                       metric,
                       d_seed_candidates.data(),
                       event_offsets.data(),
                       max_event_size,
                       blocks_per_event);
    alpaka::memcpy(queue, clue::make_host_view(seed_candidates), d_seed_candidates);
    alpaka::wait(queue);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric>
    requires(alpaka::Dim<TAcc>::value == 2)
  inline void findClusterSeedsBatched(TQueue& queue,
                                      clue::internal::SeedArray<>& seeds,
                                      PointsView<Ndim, TData>& dev_points,
                                      TData seed_dc,
                                      const DistanceMetric& metric,
                                      TData rhoc,
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
                       seed_dc,
                       metric,
                       rhoc,
                       event_associations,
                       event_offsets.data(),
                       max_event_size);
  }

}  // namespace clue::detail
