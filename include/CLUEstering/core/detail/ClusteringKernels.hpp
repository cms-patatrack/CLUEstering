
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
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData>
    requires std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>
  ALPAKA_FN_ACC void for_recursion(const TAcc& acc,
                                   std::array<int32_t, Ndim>& base_vec,
                                   const clue::SearchBoxBins<Ndim>& search_box,
                                   internal::TilesView<Ndim, TData>& tiles,
                                   PointsView<Ndim, TPointsData>& dev_points,
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

        const auto distance = [&]() -> TData {
          if constexpr (concepts::detail::view_distance_metric<DistanceMetric, Ndim>) {
            return metric(
                dev_points, static_cast<std::size_t>(point_id), static_cast<std::size_t>(j));
          } else {
            return metric(coords_i, dev_points[j]);
          }
        }();
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
              concepts::distance_metric<Ndim> DistanceMetric,
              std::floating_point TPointsData = TData>
      requires(alpaka::Dim<TAcc>::value == 1 &&
               std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TPointsData> dev_points,
                                  const KernelType& kernel,
                                  TData density_radius,
                                  DistanceMetric metric,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        auto rho_i = static_cast<TData>(0.);
        auto coords_i = dev_points[i];

        clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          const auto sigma_i = dev_points.has_sigma(dim) ? dev_points.sigma(dim)[i] : TData{0};
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
                                        i);

        assert(rho_i >= TData{0});
        dev_points.rho()[i] = rho_i;
      }
    }
  };

  struct KernelCreateFreqGrid {
    template <typename TAcc, typename TData>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TData* k_grid,  // grid of frequencies
                                  TData freq_max,
                                  TData step,  // step size between frequencies
                                  int n_grid) const {
      for (auto i : alpaka::uniformElements(acc, n_grid)) {
        k_grid[i] = -freq_max + (static_cast<TData>(i) + TData{0.5}) * step; // offset the frequency grid to avoid 0
      }
    }
  };

  struct KernelCreateMeshGrid {
    template <typename TAcc, typename TData>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TData* meshgrid,
                                  const TData* k_grid,
                                  int n_grid) const {
      for (auto idx : alpaka::uniformElements(acc, n_grid * n_grid)) {
        auto i = idx / n_grid;                        // row index
        auto j = idx % n_grid;                        //column index
        meshgrid[idx] = k_grid[i];                    // k0
        meshgrid[idx + n_grid * n_grid] = k_grid[j];  // k1
      }
    }
  };

  struct KernelFourrierTransform {
    template <typename TAcc, std::size_t Ndim,
              std::floating_point TData,
              std::floating_point TPointsData = TData>
      requires(alpaka::Dim<TAcc>::value == 1 && Ndim == 2)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const TData* meshgrid,
                                  TData* c_real,
                                  TData* c_imag,
                                  PointsView<Ndim, TPointsData> dev_points,
                                  int n_samples,
                                  int n_grid) const {
      for (auto idx : alpaka::uniformElements(acc, n_grid * n_grid)) {
        auto real = TData{};
        auto imag = TData{};

        for (auto n = 0; n < n_samples; ++n) {
          const auto x = dev_points[n][0];
          const auto y = dev_points[n][1];
          const auto k0 = meshgrid[idx];
          const auto k1 = meshgrid[idx + n_grid * n_grid];
          const auto dot_product = k0 * x + k1 * y;
          const auto weight = dev_points.weights()[n];

          real += weight * alpaka::math::cos(acc, dot_product);
          imag += weight * -1 * alpaka::math::sin(acc, dot_product);
        }

        c_real[idx] = real / n_samples;
        c_imag[idx] = imag / n_samples;
      }
    }
  };

  /*struct KernelMultByGaussian {
    template <typename TAcc, typename TData>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TData* g_hat_flat,
                                  TData* rho_hat_flat_real,
                                  TData* rho_hat_flat_imag,
                                  TData* meshgrid,
                                  TData* c_real,
                                  TData* c_imag,
                                  int n_grid,
                                  TData sigma_kernel) const {
      for (auto idx : alpaka::uniformElements(acc, n_grid * n_grid)) {
        g_hat_flat[idx] =
            math::exp(-0.5f * sigma_kernel * sigma_kernel *
                      (meshgrid[idx] * meshgrid[idx] +
                       meshgrid[idx + n_grid * n_grid] * meshgrid[idx + n_grid * n_grid]));
        rho_hat_flat_real[idx] = c_real[idx] * g_hat_flat[idx];
        rho_hat_flat_imag[idx] = c_imag[idx] * g_hat_flat[idx];
      }
    }
  };*/

    struct KernelMultByFreqResponse {
    template <typename TAcc,
              std::floating_point TData,
              concepts::convolutional_kernel KernelType>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  TData* g_hat_flat,
                                  TData* rho_hat_flat_real,
                                  TData* rho_hat_flat_imag,
                                  TData* meshgrid,
                                  TData* c_real,
                                  TData* c_imag,
                                  int n_grid,
                                  const KernelType& kernel) const {
      for (auto idx : alpaka::uniformElements(acc, n_grid * n_grid)) {
        TData k[2] = {meshgrid[idx], meshgrid[idx + n_grid * n_grid]};
        g_hat_flat[idx] = kernel.freq_eval(acc, k);
        rho_hat_flat_real[idx] = c_real[idx] * g_hat_flat[idx];
        rho_hat_flat_imag[idx] = c_imag[idx] * g_hat_flat[idx];
      }
    }
  };

  struct KernelInverseFourrierTransform {
    template <typename TAcc, std::size_t Ndim,
              std::floating_point TData,
              std::floating_point TPointsData = TData>
      requires(alpaka::Dim<TAcc>::value == 1 && Ndim == 2)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  PointsView<Ndim, TPointsData> dev_points,
                                  const TData* meshgrid,
                                  TData* rho_hat_flat_real,
                                  TData* rho_hat_flat_imag,
                                  TData prefactor,
                                  int n_samples,
                                  int n_grid) const {
      for (auto idx : alpaka::uniformElements(acc, n_samples)) {
        auto real_sum = TData{};
        for (auto i = 0; i < n_grid * n_grid; ++i) {
          auto theta =
              meshgrid[i] * dev_points[idx][0] + meshgrid[i + n_grid * n_grid] * dev_points[idx][1];
          real_sum += rho_hat_flat_real[i] * alpaka::math::cos(acc, theta) -
                      rho_hat_flat_imag[i] * alpaka::math::sin(acc, theta);
        }
        dev_points.rho()[idx] = prefactor * real_sum  * static_cast<TData>(n_samples);
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
                                                  PointsView<Ndim, TPointsData>& dev_points,
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

      for (auto tile_it = 0u; tile_it < tile_size; ++tile_it) {
        const auto j = tiles[tile_idx][tile_it];
        assert(j >= 0 && j < dev_points.size());
        auto rho_j = dev_points.rho()[j];
        bool found_higher_in_tile = (rho_j > rho_i);
        found_higher_in_tile =
            found_higher_in_tile || ((rho_j == rho_i) && (rho_j > TData{0}) && (j > point_id));

        if (found_higher_in_tile) {
          const auto distance = [&]() -> TData {
            if constexpr (concepts::detail::view_distance_metric<DistanceMetric, Ndim>) {
              return metric(
                  dev_points, static_cast<std::size_t>(point_id), static_cast<std::size_t>(j));
            } else {
              return metric(coords_i, dev_points[j]);
            }
          }();
          assert(distance >= TData{0});

          if (distance <= effective_distance && distance < delta_i) {
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
                                                         dev_points,
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
                                  internal::TilesView<Ndim, TData> dev_tiles,
                                  PointsView<Ndim, TPointsData> dev_points,
                                  TData outlier_distance,
                                  TData seeding_distance,
                                  TData min_density,
                                  DistanceMetric metric,
                                  std::size_t* seed_candidates,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        auto delta_i = std::numeric_limits<TData>::max();
        int nh_i = -1;
        auto coords_i = dev_points[i];
        auto rho_i = dev_points.rho()[i];
        const auto density_uncertainty =
            dev_points.has_uncertainty() ? dev_points.density_uncertainty()[i] : TData{1.};
        const auto effective_min_density = min_density * density_uncertainty;

        clue::SearchBoxExtremes<Ndim, TData> searchbox_extremes;
        for (auto dim = 0u; dim != Ndim; ++dim) {
          const auto sigma_i = dev_points.has_sigma(dim) ? dev_points.sigma(dim)[i] : TData{0};
          const auto box_radius =
              math::max(outlier_distance, outlier_distance * sigma_i * math::sqrt(TData{2}));
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
                                                       outlier_distance,
                                                       seeding_distance,
                                                       effective_min_density,
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
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> dev_points,
                                  TData min_density,
                                  int32_t n_points) const {
      for (auto i : alpaka::uniformElements(acc, n_points)) {
        dev_points.cluster_index()[i] = -1;
        const auto nh = dev_points.nearest_higher()[i];
        const auto rho_i = dev_points.rho()[i];
        const auto density_uncertainty =
            dev_points.has_uncertainty() ? dev_points.density_uncertainty()[i] : TData{1.};
        const auto is_seed = (nh == -1) && (rho_i >= min_density * density_uncertainty);

        if (is_seed) {
          dev_points.is_seed()[i] = 1;
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

  /*
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
                                  PointsView<Ndim, TPointsData>& dev_points,
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

*/

  /*template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData>
    requires(alpaka::Dim<TAcc>::value == 1)
  inline void computeLocalDensity(TQueue& queue,
                                  std::size_t block_size,
                                  PointsView<Ndim, TData>& dev_points,
                                  KernelType& kernel,
                                  TData freq_max,
                                  int32_t n_grid,
                                  int32_t n_points) {
    auto d_k_grid = clue::make_device_buffer<TData[]>(queue, n_grid);
    auto d_meshgrid = clue::make_device_buffer<TData[]>(queue, 2 * n_grid * n_grid);
    auto d_c_real = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_c_imag = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_rho_hat_flat_real = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_rho_hat_flat_imag = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_g_hat_flat = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);

    const auto wd_grid = clue::make_workdiv<TAcc>(nostd::ceil_div(n_grid, block_size), block_size);
    const auto wd_mesh =
        clue::make_workdiv<TAcc>(nostd::ceil_div(n_grid * n_grid, block_size), block_size);
    const auto wd_pts = clue::make_workdiv<TAcc>(nostd::ceil_div(n_points, block_size), block_size);

    auto step = TData{2} * freq_max / static_cast<TData>(n_grid);
    auto prefactor = step * step / (TData{4} * static_cast<TData>(M_PI * M_PI));

    alpaka::exec<TAcc>(
        queue, wd_grid, KernelCreateFreqGrid{}, d_k_grid.data(), freq_max, step, n_grid);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(
        queue, wd_mesh, KernelCreateMeshGrid{}, d_meshgrid.data(), d_k_grid.data(), n_grid);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(queue,
                    wd_pts,
                    KernelFourrierTransform{},
                    d_meshgrid.data(),   
                    d_c_real.data(),
                    d_c_imag.data(),
                    dev_points,          
                    n_points,
                    n_grid);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(queue,
                       wd_mesh,
                       KernelMultByGaussian{},
                       d_g_hat_flat.data(),
                       d_rho_hat_flat_real.data(),
                       d_rho_hat_flat_imag.data(),
                       d_meshgrid.data(),
                       d_c_real.data(),
                       d_c_imag.data(),
                       n_grid,
                       sigma_kernel);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(queue,
                       wd_pts,
                       KernelInverseFourrierTransform{},
                       dev_points,
                       d_meshgrid.data(),
                       d_rho_hat_flat_real.data(),
                       d_rho_hat_flat_imag.data(),
                       prefactor,
                       n_points,
                       n_grid);
    alpaka::wait(queue);
  } */


  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::convolutional_kernel KernelType>
    requires(alpaka::Dim<TAcc>::value == 1)
  inline void computeLocalDensity(TQueue& queue,
                                  std::size_t block_size,
                                  PointsView<Ndim, TData>& dev_points,
                                  const KernelType& kernel,
                                  TData freq_max,
                                  int32_t n_grid,
                                  int32_t n_points) {
    auto d_k_grid = clue::make_device_buffer<TData[]>(queue, n_grid);
    auto d_meshgrid = clue::make_device_buffer<TData[]>(queue, 2 * n_grid * n_grid);
    auto d_c_real = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_c_imag = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_rho_hat_flat_real = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_rho_hat_flat_imag = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);
    auto d_g_hat_flat = clue::make_device_buffer<TData[]>(queue, n_grid * n_grid);

    const auto wd_grid = clue::make_workdiv<TAcc>(nostd::ceil_div(n_grid, block_size), block_size);
    const auto wd_mesh =
        clue::make_workdiv<TAcc>(nostd::ceil_div(n_grid * n_grid, block_size), block_size);
    const auto wd_pts = clue::make_workdiv<TAcc>(nostd::ceil_div(n_points, block_size), block_size);

    auto step = TData{2} * freq_max / static_cast<TData>(n_grid);
    auto prefactor = step * step / (TData{4} * static_cast<TData>(M_PI * M_PI));

    alpaka::exec<TAcc>(
        queue, wd_grid, KernelCreateFreqGrid{}, d_k_grid.data(), freq_max, step, n_grid);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(
        queue, wd_mesh, KernelCreateMeshGrid{}, d_meshgrid.data(), d_k_grid.data(), n_grid);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(queue,
                    wd_pts,
                    KernelFourrierTransform{},
                    d_meshgrid.data(),
                    d_c_real.data(),
                    d_c_imag.data(),
                    dev_points,
                    n_points,
                    n_grid);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(queue,
                       wd_mesh,
                       KernelMultByFreqResponse{},
                       d_g_hat_flat.data(),
                       d_rho_hat_flat_real.data(),
                       d_rho_hat_flat_imag.data(),
                       d_meshgrid.data(),
                       d_c_real.data(),
                       d_c_imag.data(),
                       n_grid,
                       kernel);

    alpaka::wait(queue);

    alpaka::exec<TAcc>(queue,
                       wd_pts,
                       KernelInverseFourrierTransform{},
                       dev_points,
                       d_meshgrid.data(),
                       d_rho_hat_flat_real.data(),
                       d_rho_hat_flat_imag.data(),
                       prefactor,
                       n_points,
                       n_grid);
    alpaka::wait(queue);
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
                                    PointsView<Ndim, TPointsData>& dev_points,
                                    TData outlier_distance,
                                    TData seeding_distance,
                                    TData min_density,
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
                       min_density,
                       metric,
                       d_seed_candidates.data(),
                       size);
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
                               PointsView<Ndim, TData>& dev_points,
                               TData min_density,
                               int32_t size) {
    alpaka::exec<TAcc>(
        queue, work_division, KernelFindClusters{}, seeds.view(), dev_points, min_density, size);
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
