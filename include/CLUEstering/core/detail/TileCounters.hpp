
#pragma once

// Development-only counters for the tile search of the (non-batched) clustering kernels.
// Enabled by defining CLUE_TILE_COUNTERS; off by default and a no-op on the GPU backends.
// The host backends run the kernels on the calling thread, so plain thread_local increments are
// enough: no atomics, and no measurable cost per candidate.

#if defined(CLUE_TILE_COUNTERS) && !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && \
    !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
#define CLUE_TILE_COUNTERS_ACTIVE 1
#else
#define CLUE_TILE_COUNTERS_ACTIVE 0
#endif

#if CLUE_TILE_COUNTERS_ACTIVE

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace clue::detail::counters {

  struct KernelCounters {
    std::uint64_t tiles = 0;       // tiles visited
    std::uint64_t candidates = 0;  // points read from the visited tiles
    std::uint64_t distances = 0;   // distance evaluations
  };

  inline thread_local KernelCounters density;
  inline thread_local KernelCounters nearest_higher;

  // FNV-1a over the nearest-higher indices: identical clustering gives an identical hash.
  template <std::size_t Ndim, typename TData>
  inline std::uint64_t hash_nearest_higher(const PointsView<Ndim, TData>& points) {
    std::uint64_t hash = 14695981039346656037ull;
    for (auto index : points.nearest_higher()) {
      auto value = static_cast<std::uint32_t>(index);
      for (auto byte = 0; byte < 4; ++byte) {
        hash ^= (value >> (8 * byte)) & 0xffu;
        hash *= 1099511628211ull;
      }
    }
    return hash;
  }

  // With CLUE_DUMP_DIR set, writes rho and nearest_higher of every clustering call to
  // <dir>/<label>_<call>.bin as: int32 npoints, float rho[npoints], int32 nh[npoints].
  template <std::size_t Ndim, typename TPointsData>
  inline void dump(const char* label, const PointsView<Ndim, TPointsData>& points) {
    const char* dir = std::getenv("CLUE_DUMP_DIR");
    if (dir == nullptr)
      return;
    static thread_local int call = 0;
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/%s_%03d.bin", dir, label, call++);
    if (auto* file = std::fopen(path, "wb")) {
      const auto n = static_cast<std::int32_t>(points.size());
      std::fwrite(&n, sizeof(n), 1, file);
      std::fwrite(points.rho().data(), sizeof(float), n, file);
      std::fwrite(points.nearest_higher().data(), sizeof(std::int32_t), n, file);
      std::fclose(file);
    }
  }

  template <std::size_t Ndim, typename TData, typename TPointsData>
  inline void report(const internal::TilesView<Ndim, TData>& tiles,
                     const PointsView<Ndim, TPointsData>& points) {
    dump("clusterer", points);
    const auto n = static_cast<double>(points.size());
    std::fprintf(stderr, "CLUE_COUNTERS npoints=%d ntiles=%d nperdim=[", points.size(), tiles.ntiles);
    for (auto dim = 0u; dim != Ndim; ++dim)
      std::fprintf(stderr, "%s%d", dim ? "," : "", tiles.nperdim[dim]);
    std::fprintf(stderr, "] tilesize=[");
    for (auto dim = 0u; dim != Ndim; ++dim)
      std::fprintf(stderr, "%s%.3g", dim ? "," : "", static_cast<double>(tiles.tilesizes[dim]));
    std::fprintf(stderr, "]");
    const auto print = [n](const char* name, const KernelCounters& c) {
      std::fprintf(stderr,
                   " %s: tiles=%llu candidates=%llu distances=%llu per_point=[%.1f,%.1f,%.1f]",
                   name,
                   static_cast<unsigned long long>(c.tiles),
                   static_cast<unsigned long long>(c.candidates),
                   static_cast<unsigned long long>(c.distances),
                   static_cast<double>(c.tiles) / n,
                   static_cast<double>(c.candidates) / n,
                   static_cast<double>(c.distances) / n);
    };
    print("density", density);
    print("nearest_higher", nearest_higher);
    std::fprintf(stderr, " nh_hash=%016llx\n", static_cast<unsigned long long>(hash_nearest_higher(points)));
    density = {};
    nearest_higher = {};
  }

}  // namespace clue::detail::counters

#define CLUE_COUNT(counter, field, n) (clue::detail::counters::counter.field += (n))

#else

#define CLUE_COUNT(counter, field, n) (static_cast<void>(0))

#endif
