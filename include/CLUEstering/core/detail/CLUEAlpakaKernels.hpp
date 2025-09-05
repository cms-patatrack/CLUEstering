
#pragma once

#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/Tiles.hpp"
#include "CLUEstering/data_structures/internal/VecArray.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/math/math.hpp"

#include <array>
#include <alpaka/core/Common.hpp>
#include <cstdint>

namespace clue {
  namespace detail {

    using clue::PointsView;
    using clue::TilesAlpakaView;
    using clue::VecArray;

    constexpr int32_t reserve{1000000};

    template <uint8_t Ndim>
    ALPAKA_FN_ACC std::array<float, Ndim> getCoords(PointsView& d_points, int32_t i) {
      std::array<float, Ndim> coords;
      for (auto dim = 0; dim < Ndim; ++dim) {
        coords[dim] = d_points.coords[i + dim * d_points.n];
      }

      return coords;
    }

    template <typename TAcc, uint8_t Ndim, uint8_t N_, concepts::convolutional_kernel KernelType>
    ALPAKA_FN_HOST_ACC void for_recursion(const TAcc& acc,
                                          VecArray<int32_t, Ndim>& base_vec,
                                          const clue::SearchBoxBins<Ndim>& search_box,
                                          TilesAlpakaView<Ndim>& tiles,
                                          PointsView& dev_points,
                                          const KernelType& kernel,
                                          const std::array<float, Ndim>& coords_i,
                                          float* rho_i,
                                          float dc,
                                          int32_t point_id) {
      if constexpr (N_ == 0) {
        auto binId = tiles.getGlobalBinByBin(base_vec);
        auto binSize = tiles[binId].size();

        for (int binIter{}; binIter < binSize; ++binIter) {
          int32_t j{tiles[binId][binIter]};

          auto coords_j = getCoords<Ndim>(dev_points, j);
          float dist_ij_sq = tiles.distance(coords_i, coords_j);

          auto k = kernel(acc, clue::internal::math::sqrt(dist_ij_sq), point_id, j);
          *rho_i += (int)(dist_ij_sq <= dc * dc) * k * dev_points.weight[j];
        }
        return;
      } else {
        for (auto i = search_box[search_box.size() - N_][0];
             i <= search_box[search_box.size() - N_][1];
             ++i) {
          base_vec[base_vec.capacity() - N_] = i;
          for_recursion<TAcc, Ndim, N_ - 1>(
              acc, base_vec, search_box, tiles, dev_points, kernel, coords_i, rho_i, dc, point_id);
        }
      }
    }

    struct KernelCalculateLocalDensity {
      template <typename TAcc, uint8_t Ndim, concepts::convolutional_kernel KernelType>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    TilesAlpakaView<Ndim> dev_tiles,
                                    PointsView dev_points,
                                    const KernelType& kernel,
                                    float dc,
                                    int32_t n_points) const {
        for (auto i : alpaka::uniformElements(acc, n_points)) {
          float rho_i{0.f};
          auto coords_i = getCoords<Ndim>(dev_points, i);

          clue::SearchBoxExtremes<Ndim> searchbox_extremes;
          for (int dim{}; dim != Ndim; ++dim) {
            searchbox_extremes[dim] =
                clue::nostd::make_array(coords_i[dim] - dc, coords_i[dim] + dc);
          }

          clue::SearchBoxBins<Ndim> searchbox_bins;
          dev_tiles.searchBox(searchbox_extremes, searchbox_bins);

          VecArray<int32_t, Ndim> base_vec;
          for_recursion<TAcc, Ndim, Ndim>(
              acc, base_vec, searchbox_bins, dev_tiles, dev_points, kernel, coords_i, &rho_i, dc, i);

          dev_points.rho[i] = rho_i;
        }
      }
    };

    template <typename TAcc, uint8_t Ndim, uint8_t N_>
    ALPAKA_FN_HOST_ACC void for_recursion_nearest_higher(const TAcc& acc,
                                                         VecArray<int32_t, Ndim>& base_vec,
                                                         const clue::SearchBoxBins<Ndim>& search_box,
                                                         TilesAlpakaView<Ndim>& tiles,
                                                         PointsView& dev_points,
                                                         const std::array<float, Ndim>& coords_i,
                                                         float rho_i,
                                                         float* delta_i,
                                                         int* nh_i,
                                                         float dm_sq,
                                                         int32_t point_id) {
      if constexpr (N_ == 0) {
        int binId{tiles.getGlobalBinByBin(base_vec)};
        int binSize{tiles[binId].size()};

        for (int binIter{}; binIter < binSize; ++binIter) {
          const auto j{tiles[binId][binIter]};
          float rho_j{dev_points.rho[j]};
          bool found_higher{(rho_j > rho_i)};
          found_higher = found_higher || ((rho_j == rho_i) && (rho_j > 0.f) && (j > point_id));

          auto coords_j = getCoords<Ndim>(dev_points, j);

          float dist_ij_sq = tiles.distance(coords_i, coords_j);

          if (found_higher && dist_ij_sq <= dm_sq) {
            if (dist_ij_sq < *delta_i) {
              *delta_i = dist_ij_sq;
              *nh_i = j;
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
                                                           dm_sq,
                                                           point_id);
        }
      }
    }

    struct KernelCalculateNearestHigher {
      template <typename TAcc, uint8_t Ndim>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    TilesAlpakaView<Ndim> dev_tiles,
                                    PointsView dev_points,
                                    float dm,
                                    int32_t n_points) const {
        float dm_squared{dm * dm};
        for (auto i : alpaka::uniformElements(acc, n_points)) {
          float delta_i{std::numeric_limits<float>::max()};
          int nh_i{-1};
          auto coords_i = getCoords<Ndim>(dev_points, i);
          float rho_i{dev_points.rho[i]};

          clue::SearchBoxExtremes<Ndim> searchbox_extremes;
          for (int dim{}; dim != Ndim; ++dim) {
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
                                                         &delta_i,
                                                         &nh_i,
                                                         dm_squared,
                                                         i);

          dev_points.delta[i] = clue::internal::math::sqrt(delta_i);
          dev_points.nearest_higher[i] = nh_i;
        }
      }
    };

    struct KernelFindClusters {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VecArray<int32_t, reserve>* seeds,
                                    PointsView dev_points,
                                    float seed_dc,
                                    float rhoc,
                                    int32_t n_points) const {
        for (auto i : alpaka::uniformElements(acc, n_points)) {
          dev_points.cluster_index[i] = -1;

          float delta_i = dev_points.delta[i];
          float rho_i = dev_points.rho[i];

          bool is_seed = (delta_i > seed_dc) && (rho_i >= rhoc);

          if (is_seed) {
            dev_points.is_seed[i] = 1;
            seeds->push_back(acc, i);
          } else {
            dev_points.is_seed[i] = 0;
          }
        }
      }
    };

    struct KernelAssignClusters {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VecArray<int32_t, reserve>* seeds,
                                    clue::FollowersView followers,
                                    PointsView dev_points) const {
        const auto& seeds_0{*seeds};
        const auto n_seeds{seeds_0.size()};
        for (auto idx_cls : alpaka::uniformElements(acc, n_seeds)) {
          int local_stack[256] = {-1};
          int local_stack_size{};

          int idx_this_seed = seeds_0[idx_cls];
          dev_points.cluster_index[idx_this_seed] = idx_cls;
          local_stack[local_stack_size] = idx_this_seed;
          ++local_stack_size;
          while (local_stack_size > 0) {
            int idx_end_of_local_stack{local_stack[local_stack_size - 1]};
            int temp_cluster_index = dev_points.cluster_index[idx_end_of_local_stack];
            local_stack[local_stack_size - 1] = -1;
            --local_stack_size;
            const auto& followers_ies = followers[idx_end_of_local_stack];
            const auto followers_size = followers_ies.size();
            for (int j{}; j != followers_size; ++j) {
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

    template <concepts::accelerator TAcc, concepts::queue TQueue, uint8_t Ndim, typename KernelType>
    inline void computeLocalDensity(TQueue& queue,
                                    const WorkDiv& work_division,
                                    TilesAlpakaView<Ndim>& tiles,
                                    PointsView& dev_points,
                                    KernelType&& kernel,
                                    float dc,
                                    int32_t size) {
      alpaka::exec<TAcc>(queue,
                         work_division,
                         KernelCalculateLocalDensity{},
                         tiles,
                         dev_points,
                         std::forward<KernelType>(kernel),
                         dc,
                         size);
    }

    template <concepts::accelerator TAcc, concepts::queue TQueue, uint8_t Ndim>
    inline void computeNearestHighers(TQueue& queue,
                                      const WorkDiv& work_division,
                                      TilesAlpakaView<Ndim>& tiles,
                                      PointsView& dev_points,
                                      float dm,
                                      int32_t size) {
      alpaka::exec<TAcc>(
          queue, work_division, KernelCalculateNearestHigher{}, tiles, dev_points, dm, size);
    }

    template <concepts::accelerator TAcc, concepts::queue TQueue>
    inline void findClusterSeeds(TQueue& queue,
                                 const WorkDiv& work_division,
                                 VecArray<int32_t, reserve>* seeds,
                                 PointsView& dev_points,
                                 float seed_dc,
                                 float rhoc,
                                 int32_t size) {
      alpaka::exec<TAcc>(
          queue, work_division, KernelFindClusters{}, seeds, dev_points, seed_dc, rhoc, size);
    }

    template <concepts::accelerator TAcc, concepts::queue TQueue>
    inline void assignPointsToClusters(TQueue& queue,
                                       std::size_t block_size,
                                       VecArray<int32_t, reserve>* seeds,
                                       clue::FollowersView followers,
                                       PointsView dev_points) {
      const Idx grid_size = clue::divide_up_by(reserve, block_size);
      const auto work_division = clue::make_workdiv<TAcc>(grid_size, block_size);
      alpaka::exec<TAcc>(
          queue, work_division, KernelAssignClusters{}, seeds, followers, dev_points);
    }

  }  // namespace detail
}  // namespace clue
