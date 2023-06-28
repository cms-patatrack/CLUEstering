// #include "TilesAlpaka.h"
// #include "AlpakaVecArray.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  struct KernelPrepareDataStructures {
    template <typename TAcc, uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  PointsAlpaka<Ndim>& points,
                                  int n_points,
                                  TilesAlpaka<Ndim>& tiles) const {
      cms::alpakatools::for_each_element_in_grid(acc, n_points, [&](uint32_t i) {
        cms::alpakatools::VecArray<float, Ndim> coords;
        for (int j{}; j != Ndim; ++j) {
          coords[j].push_back(acc, points.coordinates_[j][i]);
        }
        tiles.fill(acc, coords, i);
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
};
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE