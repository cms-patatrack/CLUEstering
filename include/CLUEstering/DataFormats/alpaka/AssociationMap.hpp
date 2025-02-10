
#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <cassert>

#include <limits>

#include <alpaka/alpaka.hpp>

#include "../../AlpakaCore/alpakaConfig.hpp"
#include "../../AlpakaCore/alpakaMemory.hpp"
#include "../../AlpakaCore/alpakaWorkDiv.hpp"
#include "../../AlpakaCore/prefixScan.hpp"
#include "Span.hpp"

namespace clue {

  template <typename TFunc>
  struct KernelComputeAssociations {
    // TODO: implement overloads for other use-cases
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  size_t size,
                                  uint32_t* associations,
                                  TFunc func) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        associations[i] = func(acc, i);
      }
    }
  };

  struct KernelComputeAssociationSizes {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const uint32_t* associations,
                                  uint32_t* bin_sizes,
                                  size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        alpaka::atomicAdd(acc, &bin_sizes[associations[i]], 1u);
      }
    }
  };

  struct KernelFillAssociator {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  uint32_t* indexes,
                                  const uint32_t* bin_buffer,
                                  uint32_t* temp_offsets,
                                  size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        const auto binId = bin_buffer[i];
        const auto position = temp_offsets[binId];
        auto prev = alpaka::atomicAdd(acc, &temp_offsets[binId], 1u);
        indexes[prev] = i;
      }
    }
  };

  struct AssociationMapView {
    uint32_t* m_indexes;
    uint32_t* m_offsets;
    uint32_t m_nelements;
    uint32_t m_nbins;

    ALPAKA_FN_ACC Span<uint32_t> indexes(size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return Span<uint32_t>{buf_ptr, size};
    }
    ALPAKA_FN_ACC uint32_t offsets(size_t bin_id) { return m_offsets[bin_id]; }
    ALPAKA_FN_ACC Span<uint32_t> operator[](size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return Span<uint32_t>{buf_ptr, size};
    }
  };

  template <typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class AssociationMap {
  private:
    device_buffer<TDev, uint32_t[]> m_indexes;
    device_buffer<TDev, uint32_t[]> m_offsets;
    device_buffer<TDev, AssociationMapView> m_view;
    size_t m_nbins;

  public:
    AssociationMap() = default;
    // TODO: see above
    AssociationMap(size_t nelements, size_t nbins, const TDev& dev)
        : m_indexes{make_device_buffer<uint32_t[]>(dev, nelements)},
          m_offsets{make_device_buffer<uint32_t[]>(dev, nbins + 1)},
          m_view{make_device_buffer<AssociationMapView>(dev)},
          m_nbins{nbins} {
      auto h_view = make_host_buffer<AssociationMapView>(dev);
      h_view->m_indexes = m_indexes.data();
      h_view->m_offsets = m_offsets.data();
      h_view->m_size = nelements;
      h_view->m_nbins = nbins;

      auto queue(dev);
      alpaka::memcpy(queue, m_view, h_view);
      // zero the offset buffer
      alpaka::memset(queue, m_offsets, 0);
    }

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(size_t nelements, size_t nbins, TQueue queue)
        : m_indexes{make_device_buffer<uint32_t[]>(queue, nelements)},
          m_offsets{make_device_buffer<uint32_t[]>(queue, nbins + 1)},
          m_view{make_device_buffer<AssociationMapView>(queue)},
          m_nbins{nbins} {
      auto h_view = make_host_buffer<AssociationMapView>(queue);
      h_view->m_indexes = m_indexes.data();
      h_view->m_offsets = m_offsets.data();
      h_view->m_nelements = nelements;
      h_view->m_nbins = nbins;

      alpaka::memcpy(queue, m_view, h_view);
      // zero the offset buffer
      alpaka::memset(queue, m_offsets, 0);
    }

    AssociationMapView* view() { return m_view.data(); }

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    ALPAKA_FN_HOST void initialize(size_t nelements, size_t nbins, const TQueue& queue) {
      m_indexes = make_device_buffer<uint32_t[]>(queue, nelements);
      m_offsets = make_device_buffer<uint32_t[]>(queue, nbins);
      alpaka::memset(queue, m_offsets, 0);
      m_nbins = nbins;
      // TODO: I should also update the view
    }

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    ALPAKA_FN_HOST void clear(const TQueue& queue) {
      alpaka::memset(queue, m_indexes, 0);
      alpaka::memset(queue, m_offsets, 0);
      m_nbins = 0;
    }

    auto size() const { return m_nbins; }

    ALPAKA_FN_HOST device_buffer<TDev, uint32_t[]> indexes() { return m_indexes; }
    ALPAKA_FN_ACC Span<uint32_t> indexes(size_t bin_id) {
      const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
      return Span<uint32_t>{buf_ptr, size};
    }
    ALPAKA_FN_HOST device_view<TDev, uint32_t[]> indexes(const TDev& dev, size_t bin_id) {
      const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
      return make_device_view<uint32_t[], TDev>(dev, buf_ptr, size);
    }
    ALPAKA_FN_ACC Span<uint32_t> operator[](size_t bin_id) {
      const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
      return Span<uint32_t>{buf_ptr, size};
    }

    ALPAKA_FN_HOST device_buffer<TDev, uint32_t[]>& offsets() { return m_offsets; }
    ALPAKA_FN_ACC uint32_t offsets(size_t bin_id) const { return m_offsets[bin_id]; }

    template <typename TAcc,
              typename TFunc,
              typename TQueue,
              typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>,
              typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    ALPAKA_FN_HOST void fill(size_t size, TFunc func, TQueue queue) {
      auto bin_buffer = make_device_buffer<uint32_t[]>(queue, size);

      // compute associations
      const auto blocksize = 512;
      const auto gridsize = divide_up_by(size, blocksize);
      const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);
      alpaka::exec<TAcc>(queue,
                         workdiv,
                         KernelComputeAssociations<TFunc>{},
                         size,
                         bin_buffer.data(),
                         func);

      auto sizes_buffer = make_device_buffer<uint32_t[]>(queue, m_nbins);
      alpaka::memset(queue, sizes_buffer, 0);
      alpaka::exec<TAcc>(queue,
                         workdiv,
                         KernelComputeAssociationSizes{},
                         bin_buffer.data(),
                         sizes_buffer.data(),
                         size);

      // prepare prefix scan
      auto block_counter = make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, block_counter, 0);

      const auto blocksize_multiblockscan = 1024;
      auto gridsize_multiblockscan = divide_up_by(m_nbins, blocksize_multiblockscan);
      const auto workdiv_multiblockscan =
          make_workdiv<TAcc>(gridsize_multiblockscan, blocksize_multiblockscan);
      const auto dev = alpaka::getDev(queue);
      auto warp_size = alpaka::getPreferredWarpSize(dev);
      alpaka::exec<TAcc>(queue,
                         workdiv_multiblockscan,
                         multiBlockPrefixScan<uint32_t>{},
                         sizes_buffer.data(),
                         m_offsets.data() + 1,
                         m_nbins,
                         gridsize_multiblockscan,
                         block_counter.data(),
                         warp_size);

      // fill associator
      auto temp_offsets = make_device_buffer<uint32_t[]>(queue, m_nbins + 1);
      alpaka::memcpy(queue, temp_offsets, m_offsets);
      alpaka::exec<TAcc>(queue,
                         workdiv,
                         KernelFillAssociator{},
                         m_indexes.data(),
                         bin_buffer.data(),
                         temp_offsets.data(),
                         size);
    }
  };

}  // namespace clue
