
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
#include <chrono>
#include <cstdint>

namespace clue {
  namespace detail {

    using clue::PointsView;
    using clue::TilesAlpakaView;
    using clue::VecArray;

    constexpr int32_t reserve{1000000};

    template <uint8_t Ndim>
    ALPAKA_FN_ACC std::array<float, Ndim> getCoords(PointsView* d_points, int32_t i) {
      std::array<float, Ndim> coords;
      for (auto dim = 0; dim < Ndim; ++dim) {
        coords[dim] = d_points->coords[i + dim * d_points->n];
      }

      return coords;
    }

    template <typename TAcc, uint8_t Ndim, uint8_t N_, typename KernelType>
    ALPAKA_FN_HOST_ACC void for_recursion(const TAcc& acc,
                                          VecArray<int32_t, Ndim>& base_vec,
                                          const clue::SearchBoxBins<Ndim>& search_box,
                                          TilesAlpakaView<Ndim>* tiles,
                                          PointsView* dev_points,
                                          const KernelType& kernel,
                                          const std::array<float, Ndim>& coords_i,
                                          float* rho_i,
                                          float dc,
                                          int32_t point_id) {
      if constexpr (N_ == 0) {
        auto binId = tiles->getGlobalBinByBin(base_vec);
        // get the size of this bin
        auto binSize = (*tiles)[binId].size();

        // iterate inside this bin
        for (int binIter{}; binIter < binSize; ++binIter) {
          int32_t j{(*tiles)[binId][binIter]};
          // query N_{dc_}(i)

          auto coords_j = getCoords<Ndim>(dev_points, j);

          float dist_ij_sq = tiles->distance(coords_i, coords_j);

          auto k = kernel(acc, clue::internal::math::sqrt(dist_ij_sq), point_id, j);
          *rho_i += (int)(dist_ij_sq <= dc * dc) * k * dev_points->weight[j];

        }  // end of interate inside this bin
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
      template <typename TAcc, uint8_t Ndim, typename KernelType>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    TilesAlpakaView<Ndim>* dev_tiles,
                                    PointsView* dev_points,
                                    const KernelType& kernel,
                                    float dc,
                                    int32_t n_points) const {
        for (auto i : alpaka::uniformElements(acc, n_points)) {
          float rho_i{0.f};
          auto coords_i = getCoords<Ndim>(dev_points, i);

          // Get the extremes of the search box
          clue::SearchBoxExtremes<Ndim> searchbox_extremes;
          for (int dim{}; dim != Ndim; ++dim) {
            searchbox_extremes[dim] =
                clue::nostd::make_array(coords_i[dim] - dc, coords_i[dim] + dc);
          }

          // Calculate the search box
          clue::SearchBoxBins<Ndim> searchbox_bins;
          dev_tiles->searchBox(searchbox_extremes, searchbox_bins);

          VecArray<int32_t, Ndim> base_vec;
          for_recursion<TAcc, Ndim, Ndim>(
              acc, base_vec, searchbox_bins, dev_tiles, dev_points, kernel, coords_i, &rho_i, dc, i);

          dev_points->rho[i] = rho_i;
        }
      }
    };

    template <typename TAcc, uint8_t Ndim, uint8_t N_>
    ALPAKA_FN_HOST_ACC void for_recursion_nearest_higher(const TAcc& acc,
                                                         VecArray<int32_t, Ndim>& base_vec,
                                                         const clue::SearchBoxBins<Ndim>& search_box,
                                                         TilesAlpakaView<Ndim>* tiles,
                                                         PointsView* dev_points,
                                                         const std::array<float, Ndim>& coords_i,
                                                         float rho_i,
                                                         float* delta_i,
                                                         int* nh_i,
                                                         float dm_sq,
                                                         int32_t point_id) {
      if constexpr (N_ == 0) {
        int binId{tiles->getGlobalBinByBin(base_vec)};
        // get the size of this bin
        int binSize{(*tiles)[binId].size()};

        // iterate inside this bin
        for (int binIter{}; binIter < binSize; ++binIter) {
          const auto j{(*tiles)[binId][binIter]};
          // query N'_{dm}(i)
          float rho_j{dev_points->rho[j]};
          bool found_higher{(rho_j > rho_i)};
          // in the rare case where rho is the same, use detid
          found_higher = found_higher || ((rho_j == rho_i) && (rho_j > 0.f) && (j > point_id));

          // Calculate the distance between the two points
          auto coords_j = getCoords<Ndim>(dev_points, j);

          float dist_ij_sq = tiles->distance(coords_i, coords_j);

          if (found_higher && dist_ij_sq <= dm_sq) {
            // find the nearest point within N'_{dm}(i)
            if (dist_ij_sq < *delta_i) {
              // update delta_i and nearestHigher_i
              *delta_i = dist_ij_sq;
              *nh_i = j;
            }
          }
        }  // end of interate inside this bin

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
                                    TilesAlpakaView<Ndim>* dev_tiles,
                                    PointsView* dev_points,
                                    float dm,
                                    int32_t n_points) const {
        float dm_squared{dm * dm};
        for (auto i : alpaka::uniformElements(acc, n_points)) {
          float delta_i{std::numeric_limits<float>::max()};
          int nh_i{-1};
          auto coords_i = getCoords<Ndim>(dev_points, i);
          float rho_i{dev_points->rho[i]};

          // Get the extremes of the search box
          clue::SearchBoxExtremes<Ndim> searchbox_extremes;
          for (int dim{}; dim != Ndim; ++dim) {
            searchbox_extremes[dim] =
                clue::nostd::make_array(coords_i[dim] - dm, coords_i[dim] + dm);
          }

          // Calculate the search box
          clue::SearchBoxBins<Ndim> searchbox_bins;
          dev_tiles->searchBox(searchbox_extremes, searchbox_bins);

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

          dev_points->delta[i] = clue::internal::math::sqrt(delta_i);
          dev_points->nearest_higher[i] = nh_i;
        }
      }
    };

    struct KernelFindClusters {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VecArray<int32_t, reserve>* seeds,
                                    PointsView* dev_points,
                                    float seed_dc,
                                    float rhoc,
                                    int32_t n_points) const {
        for (auto i : alpaka::uniformElements(acc, n_points)) {
          // initialize cluster_index
          dev_points->cluster_index[i] = -1;

          float delta_i = dev_points->delta[i];
          float rho_i = dev_points->rho[i];

          // Determine whether the point is a seed or an outlier
          bool is_seed = (delta_i > seed_dc) && (rho_i >= rhoc);

          if (is_seed) {
            dev_points->is_seed[i] = 1;
            seeds->push_back(acc, i);
          } else {
            dev_points->is_seed[i] = 0;
          }
        }
      }
    };

    struct KernelAssignClusters {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VecArray<int32_t, reserve>* seeds,
                                    clue::FollowersView* followers,
                                    PointsView* dev_points) const {
        const auto& seeds_0{*seeds};
        const auto n_seeds{seeds_0.size()};
        for (auto idx_cls : alpaka::uniformElements(acc, n_seeds)) {
          int local_stack[256] = {-1};
          int local_stack_size{};

          int idx_this_seed = seeds_0[idx_cls];
          dev_points->cluster_index[idx_this_seed] = idx_cls;
          // push_back idThisSeed to localStack
          local_stack[local_stack_size] = idx_this_seed;
          ++local_stack_size;
          // process all elements in localStack
          while (local_stack_size > 0) {
            // get last element of localStack
            int idx_end_of_local_stack{local_stack[local_stack_size - 1]};
            int temp_cluster_index = dev_points->cluster_index[idx_end_of_local_stack];
            // pop_back last element of localStack
            local_stack[local_stack_size - 1] = -1;
            --local_stack_size;
            const auto& followers_ies = (*followers)[idx_end_of_local_stack];
            const auto followers_size = followers_ies.size();
            // loop over followers of last element of localStack
            for (int j{}; j != followers_size; ++j) {
              // pass id to follower
              int follower = followers_ies[j];
              dev_points->cluster_index[follower] = temp_cluster_index;
              // push_back follower to localStack
              local_stack[local_stack_size] = follower;
              ++local_stack_size;
            }
          }
        }
      }
    };

  }  // namespace detail
}  // namespace clue
