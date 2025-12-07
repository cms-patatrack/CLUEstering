
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/data_structures/internal/VecArray.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <array>
#include <alpaka/core/Common.hpp>
#include <cstdint>

namespace clue::detail {

  template <typename TAcc,
            std::size_t Ndim,
            std::size_t N_,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric>
  ALPAKA_FN_ACC void for_recursion(const TAcc& acc,
                                   VecArray<int32_t, Ndim>& base_vec,
                                   const clue::SearchBoxBins<Ndim>& search_box,
                                   internal::TilesView<Ndim>& tiles,
                                   PointsView<Ndim>& dev_points,
                                   const KernelType& kernel,
                                   const std::array<float, Ndim + 1>& coords_i,
                                   float& rho_i,
                                   float dc,
                                   const DistanceMetric& metric,
                                   int32_t point_id) {
    if constexpr (N_ == 0) {
      auto binId = tiles.getGlobalBinByBin(base_vec);
      auto binSize = tiles[binId].size();

      for (auto binIter = 0; binIter < binSize; ++binIter) {
        int32_t j = tiles[binId][binIter];

        auto coords_j = dev_points[j];
        auto distance = metric(coords_i, coords_j);

        auto k = kernel(acc, distance, point_id, j);
        rho_i += static_cast<int>(distance <= dc) * k * dev_points.weight[j];
      }
      return;
    } else {
      for (auto i = search_box[search_box.size() - N_][0];
           i <= search_box[search_box.size() - N_][1];
           ++i) {
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion<TAcc, Ndim, N_ - 1>(acc,
                                          base_vec,
                                          search_box,
                                          tiles,
                                          dev_points,
                                          kernel,
                                          coords_i,
                                          rho_i,
                                          dc,
                                          metric,
                                          point_id);
      }
    }
  }

  struct KernelCalculateLocalDensity {
    template <typename TAcc,
              std::size_t Ndim,
              concepts::convolutional_kernel KernelType,
              concepts::distance_metric<Ndim> DistanceMetric>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim> dev_tiles,
                                  PointsView<Ndim> dev_points,
                                  const KernelType& kernel,
                                  float dc,
                                  DistanceMetric metric,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        float rho_i = 0.f;
        auto coords_i = dev_points[i];

        clue::SearchBoxExtremes<Ndim> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          searchbox_extremes[dim] = clue::nostd::make_array(coords_i[dim] - dc, coords_i[dim] + dc);
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
                                        i);

        dev_points.rho[i] = rho_i;
      }
    }
  };

  template <typename TAcc,
            std::size_t Ndim,
            std::size_t N_,
            concepts::distance_metric<Ndim> DistanceMetric>
  ALPAKA_FN_ACC void for_recursion_nearest_higher(const TAcc& acc,
                                                  VecArray<int32_t, Ndim>& base_vec,
                                                  const clue::SearchBoxBins<Ndim>& search_box,
                                                  internal::TilesView<Ndim>& tiles,
                                                  PointsView<Ndim>& dev_points,
                                                  const std::array<float, Ndim + 1>& coords_i,
                                                  float rho_i,
                                                  float& delta_i,
                                                  int& nh_i,
                                                  float dm,
                                                  const DistanceMetric& metric,
                                                  int32_t point_id) {
    if constexpr (N_ == 0) {
      int binId = tiles.getGlobalBinByBin(base_vec);
      int binSize = tiles[binId].size();

      for (auto binIter = 0; binIter < binSize; ++binIter) {
        const auto j = tiles[binId][binIter];
        float rho_j = dev_points.rho[j];
        bool found_higher = (rho_j > rho_i);
        found_higher = found_higher || ((rho_j == rho_i) && (rho_j > 0.f) && (j > point_id));

        auto coords_j = dev_points[j];
        auto distance = metric(coords_i, coords_j);

        if (found_higher && distance <= dm) {
          if (distance < delta_i) {
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
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion_nearest_higher<TAcc, Ndim, N_ - 1>(acc,
                                                         base_vec,
                                                         search_box,
                                                         tiles,
                                                         dev_points,
                                                         coords_i,
                                                         rho_i,
                                                         delta_i,
                                                         nh_i,
                                                         dm,
                                                         metric,
                                                         point_id);
      }
    }
  }

  struct KernelCalculateNearestHigher {
    template <typename TAcc, std::size_t Ndim, concepts::distance_metric<Ndim> DistanceMetric>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim> dev_tiles,
                                  PointsView<Ndim> dev_points,
                                  float dm,
                                  DistanceMetric metric,
                                  std::size_t* seed_candidates,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        float delta_i = std::numeric_limits<float>::max();
        int nh_i = -1;
        auto coords_i = dev_points[i];
        float rho_i = dev_points.rho[i];

        clue::SearchBoxExtremes<Ndim> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          searchbox_extremes[dim] = clue::nostd::make_array(coords_i[dim] - dm, coords_i[dim] + dm);
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
                                                       metric,
                                                       i);

        dev_points.nearest_higher[i] = nh_i;
        if (nh_i == -1) {
          alpaka::atomicAdd(acc, seed_candidates, 1ul);
        }
      }
    }
  };

  struct KernelFindClusters {
    template <typename TAcc, std::size_t Ndim, concepts::distance_metric<Ndim> DistanceMetric>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim> dev_points,
                                  float seed_dc,
                                  DistanceMetric metric,
                                  float rhoc,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        dev_points.cluster_index[i] = -1;
        auto nh = dev_points.nearest_higher[i];

        auto coords_i = dev_points[i];
        auto coords_nh = dev_points[nh];
        auto distance = metric(coords_i, coords_nh);

        float rho_i = dev_points.rho[i];
        bool is_seed = (distance > seed_dc) && (rho_i >= rhoc);

        if (is_seed) {
          dev_points.is_seed[i] = 1;
          dev_points.nearest_higher[i] = -1;
          seeds.push_back(acc, i);
        } else {
          dev_points.is_seed[i] = 0;
        }
      }
    }
  };

  struct KernelAssignClusters {
    template <typename TAcc, std::size_t Ndim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  clue::FollowersView followers,
                                  PointsView<Ndim> dev_points) const {
      const auto n_seeds = seeds.size();
      for (auto idx_cls : alpaka::uniformElements(acc, n_seeds)) {
        int local_stack[256] = {-1};
        int local_stack_size = 0;

        int idx_this_seed = seeds[idx_cls];
        dev_points.cluster_index[idx_this_seed] = idx_cls;
        local_stack[local_stack_size] = idx_this_seed;
        ++local_stack_size;
        while (local_stack_size > 0) {
          int idx_end_of_local_stack = local_stack[local_stack_size - 1];
          int temp_cluster_index = dev_points.cluster_index[idx_end_of_local_stack];
          local_stack[local_stack_size - 1] = -1;
          --local_stack_size;
          const auto& followers_ies = followers[idx_end_of_local_stack];
          const auto followers_size = followers_ies.size();
          for (auto j = 0u; j != followers_size; ++j) {
            int follower = followers_ies[j];
            dev_points.cluster_index[follower] = temp_cluster_index;
            local_stack[local_stack_size] = follower;
            ++local_stack_size;
          }
        }
      }
    }
  };

  using WorkDiv = clue::WorkDiv<clue::Dim1D>;

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            concepts::convolutional_kernel KernelType,
            concepts::distance_metric<Ndim> DistanceMetric>
  inline void computeLocalDensity(TQueue& queue,
                                  const WorkDiv& work_division,
                                  internal::TilesView<Ndim>& tiles,
                                  PointsView<Ndim>& dev_points,
                                  KernelType&& kernel,
                                  float dc,
                                  const DistanceMetric& metric,
                                  int32_t size) {
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelCalculateLocalDensity{},
                       tiles,
                       dev_points,
                       std::forward<KernelType>(kernel),
                       dc,
                       metric,
                       size);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            concepts::distance_metric<Ndim> DistanceMetric>
  inline void computeNearestHighers(TQueue& queue,
                                    const WorkDiv& work_division,
                                    internal::TilesView<Ndim>& tiles,
                                    PointsView<Ndim>& dev_points,
                                    float dm,
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
                       dm,
                       metric,
                       d_seed_candidates.data(),
                       size);
    alpaka::memcpy(queue, clue::make_host_view(seed_candidates), d_seed_candidates);
    alpaka::wait(queue);
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            concepts::distance_metric<Ndim> DistanceMetric>
  inline void findClusterSeeds(TQueue& queue,
                               const WorkDiv& work_division,
                               clue::internal::SeedArray<>& seeds,
                               PointsView<Ndim>& dev_points,
                               float seed_dc,
                               const DistanceMetric& metric,
                               float rhoc,
                               int32_t size) {
    alpaka::exec<TAcc>(queue,
                       work_division,
                       KernelFindClusters{},
                       seeds.view(),
                       dev_points,
                       seed_dc,
                       metric,
                       rhoc,
                       size);
  }

  template <concepts::accelerator TAcc, concepts::queue TQueue, std::size_t Ndim>
  inline void assignPointsToClusters(TQueue& queue,
                                     std::size_t block_size,
                                     clue::internal::SeedArray<>& seeds,
                                     clue::FollowersView followers,
                                     PointsView<Ndim> dev_points) {
    const Idx grid_size = clue::divide_up_by(seeds.size(queue), block_size);
    const auto work_division = clue::make_workdiv<TAcc>(grid_size, block_size);
    alpaka::exec<TAcc>(
        queue, work_division, KernelAssignClusters{}, seeds.view(), followers, dev_points);
  }

}  // namespace clue::detail
