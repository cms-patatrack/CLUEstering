
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/meta/apply.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <alpaka/alpaka.hpp>

namespace clue::internal {

  /// @brief Grid of tiles: number of tiles and tile size along each dimension
  template <std::size_t Ndim, std::floating_point TData>
  struct TileGrid {
    std::array<int32_t, Ndim> nperdim;
    std::array<TData, Ndim> tilesizes;
    int32_t ntiles;
  };

  template <std::size_t Ndim, std::floating_point TData, clue::concepts::device TDev>
  class Tiles {
  public:
    using value_type = std::remove_cv_t<std::remove_reference_t<TData>>;

    template <clue::concepts::queue TQueue,
              clue::concepts::allocator TAllocator = clue::DefaultAllocator>
    Tiles(TQueue& queue,
          int32_t n_points,
          const TileGrid<Ndim, value_type>& grid,
          std::size_t batch_size = 1,
          const TAllocator& allocator = TAllocator{})
        : m_assoc{AssociationMap<TDev>(n_points, grid.ntiles * batch_size, queue, allocator)},
          m_minmax{make_device_buffer<CoordinateExtremes<Ndim, value_type>>(queue, allocator)},
          m_tilesizes{make_device_buffer<value_type[Ndim]>(queue, allocator)},
          m_wrapped{make_device_buffer<uint8_t[Ndim]>(queue, allocator)},
          m_ntiles{grid.ntiles},
          m_batch_size{batch_size},
          m_view{} {
      setView(n_points, grid);
    }

    const auto& view() const { return m_view; }
    auto& view() { return m_view; }

    template <clue::concepts::queue TQueue,
              clue::concepts::allocator TAllocator = clue::DefaultAllocator>
    ALPAKA_FN_HOST void initialize(TQueue& queue,
                                   int32_t npoints,
                                   const TileGrid<Ndim, value_type>& grid,
                                   std::size_t batch_size = 1,
                                   const TAllocator& allocator = TAllocator{}) {
      m_assoc.initialize(npoints, grid.ntiles * batch_size, queue, allocator);
      m_ntiles = grid.ntiles;
      m_batch_size = batch_size;
      setView(npoints, grid);
    }

    ALPAKA_FN_HOST void reset(int32_t npoints,
                              const TileGrid<Ndim, value_type>& grid,
                              std::size_t batch_size = 1) {
      m_assoc.reset(npoints, grid.ntiles * batch_size);
      m_ntiles = grid.ntiles;
      m_batch_size = batch_size;
      setView(npoints, grid);
    }

    template <typename T>
      requires std::same_as<std::remove_cv_t<T>, TData>
    struct GetGlobalBin {
      PointsView<Ndim, T> pointsView;
      TilesView<Ndim, TData> tilesView;

      ALPAKA_FN_HOST_ACC GetGlobalBin(PointsView<Ndim, T> pointsView,
                                      TilesView<Ndim, TData> tilesView)
          : pointsView(pointsView), tilesView(tilesView) {}

      ALPAKA_FN_ACC int32_t operator()(int32_t index, std::size_t event = 0) const {
        value_type coords[Ndim];
        for (auto dim = 0u; dim < Ndim; ++dim) {
          coords[dim] = pointsView.coords()[dim][index];
        }

        auto bin = tilesView.getGlobalBin(coords, event);
        return bin;
      }
    };

    template <clue::concepts::accelerator TAcc,
              clue::concepts::queue TQueue,
              std::floating_point TInput,
              clue::concepts::allocator TAllocator = clue::DefaultAllocator>
    ALPAKA_FN_HOST void fill(TQueue& queue,
                             PointsDevice<Ndim, TInput, TDev>& d_points,
                             const TAllocator& allocator = TAllocator{}) {
      auto dev = alpaka::getDev(queue);
      auto pointsView = d_points.view();
      m_assoc.template fill<TAcc>(
          d_points.size(), GetGlobalBin<TInput>(pointsView, m_view), queue, allocator);
    }

    template <clue::concepts::accelerator TAcc,
              clue::concepts::queue TQueue,
              std::floating_point TInput,
              clue::concepts::allocator TAllocator = clue::DefaultAllocator>
    ALPAKA_FN_HOST void fill_batch(TQueue& queue,
                                   PointsDevice<Ndim, TInput, TDev>& d_points,
                                   const auto& event_offsets,
                                   std::size_t max_event_size,
                                   const TAllocator& allocator = TAllocator{}) {
      auto dev = alpaka::getDev(queue);
      auto pointsView = d_points.view();
      m_assoc.template fill_batch<TAcc>(queue,
                                        d_points.size(),
                                        GetGlobalBin<TInput>(pointsView, m_view),
                                        event_offsets,
                                        max_event_size,
                                        allocator);
    }

    ALPAKA_FN_HOST inline clue::device_buffer<TDev, CoordinateExtremes<Ndim, value_type>> minMax()
        const {
      return m_minmax;
    }
    ALPAKA_FN_HOST inline clue::device_buffer<TDev, value_type[Ndim]> tileSize() const {
      return m_tilesizes;
    }
    ALPAKA_FN_HOST inline clue::device_buffer<TDev, uint8_t[Ndim]> wrapped() const {
      return m_wrapped;
    }

    ALPAKA_FN_HOST inline constexpr auto size() const { return m_ntiles; }

    ALPAKA_FN_HOST inline constexpr const auto& nPerDim() const { return m_view.nperdim; }

    ALPAKA_FN_HOST inline constexpr auto extents() const { return m_assoc.extents(); }

  private:
    ALPAKA_FN_HOST void setView(int32_t npoints, const TileGrid<Ndim, value_type>& grid) {
      m_view.indexes = m_assoc.indexes().data();
      m_view.offsets = m_assoc.offsets().data();
      m_view.minmax = m_minmax.data();
      m_view.tilesizes = m_tilesizes.data();
      m_view.wrapping = m_wrapped.data();
      m_view.npoints = npoints;
      m_view.ntiles = grid.ntiles;
      m_view.nperdim = grid.nperdim;
      // row-major: the last dimension varies fastest
      int32_t stride = 1;
      meta::apply<Ndim>([&]<std::size_t Id>() {
        constexpr auto dim = Ndim - 1 - Id;
        m_view.strides[dim] = stride;
        stride *= grid.nperdim[dim];
      });
    }

    AssociationMap<TDev> m_assoc;
    device_buffer<TDev, CoordinateExtremes<Ndim, value_type>> m_minmax;
    device_buffer<TDev, value_type[Ndim]> m_tilesizes;
    device_buffer<TDev, uint8_t[Ndim]> m_wrapped;
    int32_t m_ntiles;
    std::size_t m_batch_size;
    TilesView<Ndim, value_type> m_view;
  };

}  // namespace clue::internal
