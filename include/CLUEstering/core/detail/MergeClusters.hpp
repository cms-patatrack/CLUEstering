#pragma once

// Optional post-processing step of CLUE: persistence-based merging of the clusters (ToMATo-like).
//
// Two clusters a and b are merged when the density saddle between them is high compared to their
// peaks:
//     saddle(a, b) >= threshold * min(peak_a, peak_b)
// where saddle(a, b) = max over neighbouring pairs (i in a, j in b, d(i, j) <= density_radius) of
// min(rho_i, rho_j), and peak_c is the density of the seed of c (the maximum density in c).
// Merges are applied in order of decreasing saddle with a union-find; the merged cluster keeps
// the higher peak. The result is always a coarsening of the plain CLUE partition, outliers stay
// outliers.
//
// Device side: one thread per point computes the saddles with the neighbouring clusters and emits
// one edge per (point, neighbouring cluster). Host side: reduction of the edges by cluster pair,
// union-find, relabelling. The number of edges is of the order of the number of points on the
// cluster boundaries, and the number of clusters is small, so the host part is negligible.

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/internal/SearchBox.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/alpaka/default_allocator.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/math/math.hpp"
#include "CLUEstering/internal/nostd/ceil_div.hpp"

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

CLUE_NAMESPACE_BEGIN
namespace detail {

  template <std::floating_point TData>
  struct SaddleEdge {
    int32_t a;  // smaller cluster index
    int32_t b;  // larger cluster index
    TData saddle;
  };

  // Per-thread accumulator of the saddles between the cluster of point i and the neighbouring
  // clusters. A point on a boundary usually touches one or two other clusters, so a small
  // fixed-size buffer in registers is enough; when it is full the edges are emitted directly.
  template <std::floating_point TData, std::size_t Capacity = 8>
  struct LocalSaddles {
    int32_t cluster[Capacity];
    TData saddle[Capacity];
    std::size_t size = 0;
  };

  template <typename TAcc, std::floating_point TData>
  ALPAKA_FN_ACC inline void emit_edge(const TAcc& acc,
                                      SaddleEdge<TData>* edges,
                                      std::size_t* n_edges,
                                      std::size_t capacity,
                                      int32_t a,
                                      int32_t b,
                                      TData saddle) {
    const auto idx = alpaka::atomicAdd(acc, n_edges, std::size_t{1});
    // when the buffer is too small the counter keeps growing: the host reallocates and reruns
    if (idx < capacity) {
      edges[idx] = SaddleEdge<TData>{a, b, saddle};
    }
  }

  template <typename TAcc,
            std::size_t Ndim,
            std::size_t N_,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData,
            std::size_t LocalCapacity>
    requires std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>
  ALPAKA_FN_ACC void for_recursion_saddles(const TAcc& acc,
                                           std::array<int32_t, Ndim>& base_vec,
                                           const clue::SearchBoxBins<Ndim>& search_box,
                                           internal::TilesView<Ndim, TData>& tiles,
                                           PointsView<Ndim, TPointsData>& points,
                                           const std::array<TData, Ndim + 1>& coords_i,
                                           int32_t point_id,
                                           int32_t cluster_i,
                                           TData rho_i,
                                           TData rho_threshold,
                                           TData density_radius,
                                           const DistanceMetric& metric,
                                           LocalSaddles<TData, LocalCapacity>& local,
                                           SaddleEdge<TData>* edges,
                                           std::size_t* n_edges,
                                           std::size_t capacity) {
    if constexpr (N_ == 0) {
      const auto tile_idx = tiles.getGlobalBinByBin(base_vec, 0);
      const auto tile_size = tiles[tile_idx].size();

      for (auto tile_it = 0u; tile_it < tile_size; ++tile_it) {
        const auto j = tiles[tile_idx][tile_it];
        assert(j >= 0 && j < points.size());

        // each unordered pair of points is visited once, from the cluster with the smaller index;
        // this also skips the outliers (-1) and the points of the same cluster
        const auto cluster_j = points.cluster_index()[j];
        if (cluster_j <= cluster_i)
          continue;
        const auto rho_j = points.rho()[j];
        if (rho_j < rho_threshold)
          continue;

        const auto distance = [&]() -> TData {
          if constexpr (concepts::detail::view_distance_metric<DistanceMetric, Ndim>) {
            return metric(points, static_cast<std::size_t>(point_id), static_cast<std::size_t>(j));
          } else {
            return metric(coords_i, points[j]);
          }
        }();
        if (distance > density_radius)
          continue;

        const auto saddle = math::min(rho_i, rho_j);
        auto found = false;
        for (auto k = 0u; k < local.size; ++k) {
          if (local.cluster[k] == cluster_j) {
            local.saddle[k] = math::max(local.saddle[k], saddle);
            found = true;
            break;
          }
        }
        if (!found) {
          if (local.size < LocalCapacity) {
            local.cluster[local.size] = cluster_j;
            local.saddle[local.size] = saddle;
            ++local.size;
          } else {
            emit_edge(acc, edges, n_edges, capacity, cluster_i, cluster_j, saddle);
          }
        }
      }
    } else {
      for (auto i = search_box[search_box.size() - N_][0];
           i <= search_box[search_box.size() - N_][1];
           ++i) {
        base_vec[Ndim - N_] = i;
        for_recursion_saddles<TAcc, Ndim, N_ - 1>(acc,
                                                  base_vec,
                                                  search_box,
                                                  tiles,
                                                  points,
                                                  coords_i,
                                                  point_id,
                                                  cluster_i,
                                                  rho_i,
                                                  rho_threshold,
                                                  density_radius,
                                                  metric,
                                                  local,
                                                  edges,
                                                  n_edges,
                                                  capacity);
      }
    }
  }

  struct KernelComputeSaddles {
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
                                  TData density_radius,
                                  TData rho_threshold,
                                  DistanceMetric metric,
                                  SaddleEdge<TData>* edges,
                                  std::size_t* n_edges,
                                  std::size_t capacity) const {
      for (auto i : alpaka::uniformElements(acc, points.size())) {
        const auto cluster_i = points.cluster_index()[i];
        const auto rho_i = points.rho()[i];
        // a saddle through i is at most rho_i: if it is below threshold * (smallest peak) it can
        // never cause a merge, so most of the points in the cluster tails are skipped
        if (cluster_i < 0 || rho_i < rho_threshold)
          continue;

        const auto coords_i = points[i];
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

        LocalSaddles<TData> local;
        std::array<int32_t, Ndim> base_vec{};
        for_recursion_saddles<TAcc, Ndim, Ndim>(acc,
                                                base_vec,
                                                searchbox_bins,
                                                tiles,
                                                points,
                                                coords_i,
                                                static_cast<int32_t>(i),
                                                cluster_i,
                                                rho_i,
                                                rho_threshold,
                                                density_radius,
                                                metric,
                                                local,
                                                edges,
                                                n_edges,
                                                capacity);
        for (auto k = 0u; k < local.size; ++k) {
          emit_edge(acc, edges, n_edges, capacity, cluster_i, local.cluster[k], local.saddle[k]);
        }
      }
    }
  };

  struct KernelGatherPeaks {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  clue::internal::SeedArrayView seeds,
                                  PointsView<Ndim, TData> points,
                                  std::remove_cv_t<TData>* peaks) const {
      // the seed of cluster c is seeds[c] (see KernelAssignSeedIndices) and has the maximum
      // density of its cluster, because the nearest-higher chains are increasing in density
      for (auto c : alpaka::uniformElements(acc, seeds.size())) {
        peaks[c] = points.rho()[seeds[c]];
      }
    }
  };

  struct KernelRelabelClusters {
    template <typename TAcc, std::size_t Ndim, std::floating_point TData>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  PointsView<Ndim, TData> points,
                                  const int32_t* new_label,
                                  const int32_t* absorbed) const {
      for (auto i : alpaka::uniformElements(acc, points.size())) {
        const auto c = points.cluster_index()[i];
        if (c < 0)
          continue;
        points.cluster_index()[i] = new_label[c];
        // the seeds of the absorbed clusters are not seeds anymore
        if (points.is_seed()[i] && absorbed[c])
          points.is_seed()[i] = 0;
      }
    }
  };

  // Host side: reduce the edges by cluster pair, union-find in order of decreasing saddle.
  // Returns the new label of each old cluster (compacted to 0..K'-1, in order of the surviving
  // seed) and whether each old cluster was absorbed by another one.
  template <std::floating_point TData>
  inline void merge_cluster_graph(std::vector<SaddleEdge<TData>>& edges,
                                  const std::vector<TData>& peaks,
                                  TData threshold,
                                  std::vector<int32_t>& new_label,
                                  std::vector<int32_t>& absorbed) {
    const auto n_clusters = peaks.size();

    // max saddle per pair: sort by pair, then keep the first of each run
    std::ranges::sort(edges, [](const auto& x, const auto& y) {
      return x.a != y.a ? x.a < y.a : x.b != y.b ? x.b < y.b : x.saddle > y.saddle;
    });
    const auto [last, end] = std::ranges::unique(
        edges, [](const auto& x, const auto& y) { return x.a == y.a && x.b == y.b; });
    edges.erase(last, end);
    // decreasing saddle; ties broken by the pair to make the result deterministic
    std::ranges::sort(edges, [](const auto& x, const auto& y) {
      return x.saddle != y.saddle ? x.saddle > y.saddle : (x.a != y.a ? x.a < y.a : x.b < y.b);
    });

    std::vector<int32_t> parent(n_clusters);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&parent](int32_t c) {
      while (parent[c] != c) {
        parent[c] = parent[parent[c]];
        c = parent[c];
      }
      return c;
    };
    // after a merge the root keeps the higher peak, so peaks[root] is the peak of the union
    for (const auto& e : edges) {
      const auto ra = find(e.a);
      const auto rb = find(e.b);
      if (ra == rb)
        continue;
      if (e.saddle >= threshold * std::min(peaks[ra], peaks[rb])) {
        const bool a_higher = peaks[ra] > peaks[rb] || (peaks[ra] == peaks[rb] && ra < rb);
        if (a_higher)
          parent[rb] = ra;
        else
          parent[ra] = rb;
      }
    }

    new_label.assign(n_clusters, -1);
    absorbed.assign(n_clusters, 0);
    int32_t next = 0;
    for (auto c = 0u; c < n_clusters; ++c) {
      if (find(c) == static_cast<int32_t>(c))
        new_label[c] = next++;
    }
    for (auto c = 0u; c < n_clusters; ++c) {
      const auto root = find(c);
      new_label[c] = new_label[root];
      absorbed[c] = (root != static_cast<int32_t>(c));
    }
  }

  template <concepts::accelerator TAcc,
            concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TData,
            concepts::distance_metric<Ndim> DistanceMetric,
            std::floating_point TPointsData = TData,
            concepts::allocator TAllocator = DefaultAllocator>
    requires(alpaka::Dim<TAcc>::value == 1 &&
             std::same_as<std::remove_cv_t<TPointsData>, std::remove_cv_t<TData>>)
  inline void mergeClusters(TQueue& queue,
                            std::size_t block_size,
                            internal::TilesView<Ndim, TData>& tiles,
                            PointsView<Ndim, TPointsData>& points,
                            clue::internal::SeedArray<>& seeds,
                            TData threshold,
                            TData density_radius,
                            const DistanceMetric& metric,
                            const TAllocator& allocator = TAllocator{}) {
    const auto n_clusters = seeds.size(queue);
    if (n_clusters < 2)
      return;

    // 1. peaks of the clusters
    auto d_peaks = internal::alloc_device_buffer<TData>(queue, n_clusters, allocator);
    alpaka::exec<TAcc>(
        queue,
        clue::make_workdiv<TAcc>(nostd::ceil_div(n_clusters, block_size), block_size),
        KernelGatherPeaks{},
        seeds.view(),
        points,
        d_peaks.data());
    std::vector<TData> peaks(n_clusters);
    alpaka::memcpy(queue, alpaka::createView(host, peaks.data(), n_clusters), d_peaks);
    alpaka::wait(queue);
    const auto rho_threshold = threshold * *std::ranges::min_element(peaks);

    // 2. saddles. The buffer is sized for a fraction of the points; if it overflows the kernel
    // is rerun with the exact size (the counter keeps counting past the capacity).
    const Idx point_grid = nostd::ceil_div(points.size(), block_size);
    const auto point_workdiv = clue::make_workdiv<TAcc>(point_grid, block_size);
    auto d_n_edges = internal::alloc_device_buffer<std::size_t>(queue, Scalar{}, allocator);
    std::size_t capacity = std::max<std::size_t>(1024, points.size() / 4);
    std::size_t n_edges = 0;
    std::vector<SaddleEdge<TData>> edges;
    while (true) {
      auto d_edges = internal::alloc_device_buffer<SaddleEdge<TData>>(queue, capacity, allocator);
      alpaka::memset(queue, d_n_edges, 0u);
      alpaka::exec<TAcc>(queue,
                         point_workdiv,
                         KernelComputeSaddles{},
                         tiles,
                         points,
                         density_radius,
                         rho_threshold,
                         metric,
                         d_edges.data(),
                         d_n_edges.data(),
                         capacity);
      alpaka::memcpy(queue, alpaka::createView(host, &n_edges, Scalar{}), d_n_edges);
      alpaka::wait(queue);
      if (n_edges <= capacity) {
        edges.resize(n_edges);
        if (n_edges > 0) {
          alpaka::memcpy(queue,
                         alpaka::createView(host, edges.data(), n_edges),
                         alpaka::createView(alpaka::getDev(queue), d_edges.data(), n_edges));
          alpaka::wait(queue);
        }
        break;
      }
      capacity = n_edges;
    }
    if (edges.empty())
      return;

    // 3. union-find on the host
    std::vector<int32_t> new_label, absorbed;
    merge_cluster_graph(edges, peaks, threshold, new_label, absorbed);

    // 4. relabel the points
    auto d_new_label = internal::alloc_device_buffer<int32_t>(queue, n_clusters, allocator);
    auto d_absorbed = internal::alloc_device_buffer<int32_t>(queue, n_clusters, allocator);
    alpaka::memcpy(queue, d_new_label, alpaka::createView(host, new_label.data(), n_clusters));
    alpaka::memcpy(queue, d_absorbed, alpaka::createView(host, absorbed.data(), n_clusters));
    alpaka::exec<TAcc>(queue,
                       point_workdiv,
                       KernelRelabelClusters{},
                       points,
                       d_new_label.data(),
                       d_absorbed.data());
    alpaka::wait(queue);
    // TODO: the SeedArray still contains the seeds of the absorbed clusters, so getSeeds()
    // returns the peaks before the merge. Compact it here if getSeeds() must reflect the merge.
  }

}  // namespace detail
CLUE_NAMESPACE_END
