#include <alpaka/core/Common.hpp>
#include <cstdint>

#include "../AlpakaCore/alpakaWorkDiv.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "../DataFormats/alpaka/TilesAlpaka.h"
#include "../DataFormats/alpaka/AlpakaVecArray.h"

using cms::alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <uint8_t Ndim>
  using PointsView = typename PointsAlpaka<Ndim>::PointsAlpakaView;

  struct KernelPrepareDataStructures {
    template <typename TAcc, uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  PointsAlpaka<Ndim>* points,
                                  TilesAlpaka<TAcc, Ndim>* tiles,
                                  uint32_t n_points) const {
      cms::alpakatools::for_each_element_in_grid(
          acc, n_points, [&](uint32_t i) { tiles->fill(acc, points->coords[i], i); });
    }
  };

  template <typename TAcc, uint8_t Ndim, uint8_t N_>
  ALPAKA_FN_HOST_ACC void for_recursion(VecArray<uint32_t, Ndim>& base_vec,
                                        const VecArray<VecArray<uint32_t, 2>, Ndim>& search_box,
                                        TilesAlpaka<TAcc, Ndim>* tiles,
                                        PointsView<Ndim>* dev_points,
                                        const VecArray<float, Ndim>& point_coordinates,
                                        float* rho_i,
                                        float dc,
                                        int i) {
    if constexpr (N_ == 0) {
      int binId{tiles->getGlobalBinByBin(base_vec)};
      // get the size of this bin
      int binSize{static_cast<int>(tiles[binId].size())};

      // iterate inside this bin
      for (int binIter{}; binIter < binSize; ++binIter) {
        int j{tiles[binId][binIter]};
        // query N_{dc_}(i)

        VecArray<float, Ndim> j_coords{dev_points[j]};
        float dist_ij_sq{0.f};
        for (int dim{}; dim != Ndim; ++dim) {
          dist_ij_sq += (j_coords[dim] - point_coordinates[dim]) * (j_coords[dim] - point_coordinates[dim]);
        }

        if (dist_ij_sq <= dc * dc) {
          *rho_i += (i == j ? 1.f : 0.5f);
        }
      }  // end of interate inside this bin

      return;
    } else {
      for (int i{search_box[search_box.size() - N_][0]}; i <= search_box[search_box.size() - N_][1]; ++i) {
        base_vec[base_vec.size() - N_] = i;
        for_recursion<TAcc, Ndim, N_ - 1>(base_vec, search_box, tiles, dev_points, point_coordinates, rho_i, i);
      }
    }
  }

  struct KernelCalculateLocalDensity {
    template <typename TAcc, uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TilesAlpaka<TAcc, Ndim>* dev_tiles,
                                  PointsView<Ndim>* dev_points,
                                  float dc,
                                  uint32_t n_points) {
      const float dc_squared{dc * dc};
      cms::alpakatools::for_each_element_in_grid(acc, n_points, [&](uint32_t i) {
        float rho_i{0.f};
        VecArray<float, Ndim> point_coordinates{dev_points.coords[i]};

        VecArray<VecArray<float, 2>, Ndim> searchbox_extremes;
        for (int dim{}; dim != Ndim; ++dim) {
          VecArray<float, 2> dim_extremes;
          dim_extremes.push_back(acc, point_coordinates[dim] - dc);
          dim_extremes.push_back(acc, point_coordinates[dim] + dc);

          searchbox_extremes.push_back(acc, dim_extremes);
        }

        VecArray<VecArray<uint32_t, 2>, Ndim> search_box = dev_tiles->searchBox(searchbox_extremes);

        VecArray<uint32_t, Ndim> base_vec;
        for_recursion<TAcc, Ndim, Ndim>(base_vec, search_box, dev_tiles, dev_points, point_coordinates, &rho_i, dc, i);
      });
    }
  };

  // struct KernelFindClusters {
  //   template <typename TAcc, uint8_t Ndim>
  //   ALPAKA_FN_ACC void operator()(const TAcc& acc, Queue queue_, PoitsAlpaka<Ndim> points, int n_points) const {
  //     int n_clusters{};
  //     uint32_t j{};
  //     auto local_stack = cms::alpakatools::make_device_buffer<Device, uint32_t[]>(queue_, n_points);
  //     cms::alpakatools::for_each_element_in_grid(acc, n_points, [&](uint32_t i) {
  //       poits->cluster_index[i] = -1;
  //       float delta_i{points->delta[i]};
  //       float rho_i{points->rho[i]};

  //       // determine seed or outlier
  //       bool is_seed{(deltai > dc_) && (rhoi >= rhoc_)};
  //       bool is_outlier{(deltai > outlierDeltaFactor_ * dc_) && (rhoi < rhoc_)};

  //       if (is_seed) {
  //         points->is_seed[i] = 1;
  //         points->cluster_index[i] = n_clusters;
  //         ++n_clusters;
  //         local_stack[j] = i;
  //         ++j;
  //       } else if (!is_outlier) {
  //         points->followers[points->nearest_higher[i]] = i;
  //       }
  //     });
  //   }
  // };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
