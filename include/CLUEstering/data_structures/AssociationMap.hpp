
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

#include "CLUEstering/data_structures/internal/Span.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/algorithm/scan/scan.hpp"

namespace clue {

  namespace concepts = detail::concepts;

  template <typename TFunc>
  struct KernelComputeAssociations {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  size_t size,
                                  int32_t* associations,
                                  TFunc func) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        associations[i] = func(i);
      }
    }
  };

  struct KernelComputeAssociationSizes {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const int32_t* associations,
                                  int32_t* bin_sizes,
                                  size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        if (associations[i] >= 0)
          alpaka::atomicAdd(acc, &bin_sizes[associations[i]], 1);
      }
    }
  };

  struct KernelFillAssociator {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  int32_t* indexes,
                                  const int32_t* bin_buffer,
                                  int32_t* temp_offsets,
                                  size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        const auto binId = bin_buffer[i];
        if (binId >= 0) {
          auto prev = alpaka::atomicAdd(acc, &temp_offsets[binId], 1);
          indexes[prev] = i;
        }
      }
    }
  };

  struct AssociationMapView {
    int32_t* m_indexes;
    int32_t* m_offsets;
    int32_t m_nelements;
    int32_t m_nbins;

    ALPAKA_FN_ACC Span<int32_t> indexes(size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return Span<int32_t>{buf_ptr, size};
    }
    ALPAKA_FN_ACC int32_t offsets(size_t bin_id) { return m_offsets[bin_id]; }
    ALPAKA_FN_ACC Span<int32_t> operator[](size_t bin_id) {
      auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes + m_offsets[bin_id];
      return Span<int32_t>{buf_ptr, size};
    }
  };

  template <concepts::device TDev>
  class AssociationMap {
  public:
    using key_type = int32_t;
    using mapped_type = int32_t;
    using size_type = std::size_t;

    struct Extents {
      size_type content;
      size_type offset;
    };

    AssociationMap() = default;
    // TODO: see above
    AssociationMap(size_type nelements, size_type nbins, const TDev& dev)
        : m_indexes{make_device_buffer<mapped_type[]>(dev, nelements)},
          m_offsets{make_device_buffer<key_type[]>(dev, nbins + 1)},
          m_hview{make_host_buffer<AssociationMapView>(dev)},
          m_view{make_device_buffer<AssociationMapView>(dev)},
          m_nbins{nbins} {
      m_hview->m_indexes = m_indexes.data();
      m_hview->m_offsets = m_offsets.data();
      m_hview->m_nelements = nelements;
      m_hview->m_nbins = nbins;

      auto queue(dev);
      alpaka::memcpy(queue, m_view, m_hview);
      // zero the offset buffer
      alpaka::memset(queue, m_offsets, 0);
    }

    template <concepts::queue TQueue>
    AssociationMap(size_type nelements, size_type nbins, TQueue& queue)
        : m_indexes{make_device_buffer<mapped_type[]>(queue, nelements)},
          m_offsets{make_device_buffer<key_type[]>(queue, nbins + 1)},
          m_hview{make_host_buffer<AssociationMapView>(queue)},
          m_view{make_device_buffer<AssociationMapView>(queue)},
          m_nbins{nbins} {
      m_hview->m_indexes = m_indexes.data();
      m_hview->m_offsets = m_offsets.data();
      m_hview->m_nelements = nelements;
      m_hview->m_nbins = nbins;

      alpaka::memcpy(queue, m_view, m_hview);
      // zero the offset buffer
      alpaka::memset(queue, m_offsets, 0);
    }

    AssociationMapView* view() { return m_view.data(); }

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void initialize(size_type nelements, size_type nbins, TQueue& queue) {
      m_indexes = make_device_buffer<int32_t[]>(queue, nelements);
      m_offsets = make_device_buffer<int32_t[]>(queue, nbins);
      alpaka::memset(queue, m_offsets, 0);
      m_nbins = nbins;

      m_hview->m_indexes = m_indexes.data();
      m_hview->m_offsets = m_offsets.data();
      m_hview->m_nelements = nelements;
      m_hview->m_nbins = nbins;
      alpaka::memcpy(queue, m_view, m_hview);
    }

    template <concepts::queue TQueue>
    ALPAKA_FN_HOST void reset(TQueue& queue, size_type nelements, size_type nbins) {
      alpaka::memset(queue, m_indexes, 0);
      alpaka::memset(queue, m_offsets, 0);
      m_nbins = nbins;

      m_hview->m_indexes = m_indexes.data();
      m_hview->m_offsets = m_offsets.data();
      m_hview->m_nelements = nelements;
      m_hview->m_nbins = nbins;
      alpaka::memcpy(queue, m_view, m_hview);
    }

    auto size() const { return m_nbins; }

    auto extents() const {
      return Extents{
          alpaka::trait::GetExtents<clue::device_buffer<TDev, mapped_type[]>>{}(m_indexes)[0u],
          alpaka::trait::GetExtents<clue::device_buffer<TDev, key_type[]>>{}(m_offsets)[0u]};
    }

    ALPAKA_FN_HOST const device_buffer<TDev, mapped_type[]>& indexes() const { return m_indexes; }
    ALPAKA_FN_HOST device_buffer<TDev, mapped_type[]>& indexes() { return m_indexes; }

    ALPAKA_FN_ACC Span<int32_t> indexes(size_type bin_id) {
      const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
      return Span<mapped_type>{buf_ptr, size};
    }
    ALPAKA_FN_HOST device_view<TDev, int32_t[]> indexes(const TDev& dev, size_type bin_id) {
      const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
      return make_device_view<int32_t[], TDev>(dev, buf_ptr, size);
    }
    ALPAKA_FN_ACC Span<int32_t> operator[](size_type bin_id) {
      const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
      return Span<int32_t>{buf_ptr, size};
    }

    ALPAKA_FN_HOST const device_buffer<TDev, int32_t[]>& offsets() const { return m_offsets; }
    ALPAKA_FN_HOST device_buffer<TDev, int32_t[]>& offsets() { return m_offsets; }

    ALPAKA_FN_ACC int32_t offsets(size_type bin_id) const { return m_offsets[bin_id]; }

    template <concepts::accelerator TAcc, typename TFunc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, TFunc func, TQueue& queue) {
      auto bin_buffer = make_device_buffer<int32_t[]>(queue, size);

      // compute associations
      const auto blocksize = 512;
      const auto gridsize = divide_up_by(size, blocksize);
      const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);
      alpaka::exec<TAcc>(
          queue, workdiv, KernelComputeAssociations<TFunc>{}, size, bin_buffer.data(), func);

      auto sizes_buffer = make_device_buffer<int32_t[]>(queue, m_nbins);
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
                         multiBlockPrefixScan<int32_t>{},
                         sizes_buffer.data(),
                         m_offsets.data() + 1,
                         m_nbins,
                         gridsize_multiblockscan,
                         block_counter.data(),
                         warp_size);

      // fill associator
      auto temp_offsets = make_device_buffer<int32_t[]>(queue, m_nbins + 1);
      alpaka::memcpy(queue,
                     temp_offsets,
                     make_device_view(alpaka::getDev(queue), m_offsets.data(), m_nbins + 1));
      alpaka::exec<TAcc>(queue,
                         workdiv,
                         KernelFillAssociator{},
                         m_indexes.data(),
                         bin_buffer.data(),
                         temp_offsets.data(),
                         size);
    }

    template <concepts::accelerator TAcc, concepts::queue TQueue>
    ALPAKA_FN_HOST void fill(size_type size, std::span<key_type> associations, TQueue& queue) {
      const auto blocksize = 512;
      const auto gridsize = divide_up_by(size, blocksize);
      const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);

      auto sizes_buffer = make_device_buffer<key_type[]>(queue, m_nbins);
      alpaka::memset(queue, sizes_buffer, 0);
      alpaka::exec<TAcc>(queue,
                         workdiv,
                         KernelComputeAssociationSizes{},
                         associations.data(),
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
                         multiBlockPrefixScan<key_type>{},
                         sizes_buffer.data(),
                         m_offsets.data() + 1,
                         m_nbins,
                         gridsize_multiblockscan,
                         block_counter.data(),
                         warp_size);

      // fill associator
      auto temp_offsets = make_device_buffer<key_type[]>(queue, m_nbins + 1);
      alpaka::memcpy(queue,
                     temp_offsets,
                     make_device_view(alpaka::getDev(queue), m_offsets.data(), m_nbins + 1));
      alpaka::exec<TAcc>(queue,
                         workdiv,
                         KernelFillAssociator{},
                         m_indexes.data(),
                         associations.data(),
                         temp_offsets.data(),
                         size);
    }

  private:
    device_buffer<TDev, mapped_type[]> m_indexes;
    device_buffer<TDev, key_type[]> m_offsets;
    host_buffer<AssociationMapView> m_hview;
    device_buffer<TDev, AssociationMapView> m_view;
    size_type m_nbins;
  };

}  // namespace clue
