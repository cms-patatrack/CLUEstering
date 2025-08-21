
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/internal/Span.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/internal/algorithm/scan/scan.hpp"

#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  namespace detail {

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

  }  // namespace detail

  template <concepts::device TDev>
  inline AssociationMap<TDev>::AssociationMap(size_type nelements, size_type nbins, const TDev& dev)
      : m_indexes{make_device_buffer<mapped_type[]>(dev, nelements)},
        m_offsets{make_device_buffer<key_type[]>(dev, nbins + 1)},
        m_view{},
        m_nbins{nbins} {
    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_nelements = nelements;
    m_view.m_nbins = nbins;

    auto queue(dev);
    // zero the offset buffer
    alpaka::memset(queue, m_offsets, 0);
  }

  template <concepts::device TDev>
  template <concepts::queue TQueue>
  inline AssociationMap<TDev>::AssociationMap(size_type nelements, size_type nbins, TQueue& queue)
      : m_indexes{make_device_buffer<mapped_type[]>(queue, nelements)},
        m_offsets{make_device_buffer<key_type[]>(queue, nbins + 1)},
        m_view{},
        m_nbins{nbins} {
    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_nelements = nelements;
    m_view.m_nbins = nbins;

    // zero the offset buffer
    alpaka::memset(queue, m_offsets, 0);
  }

  template <concepts::device TDev>
  inline auto AssociationMap<TDev>::extents() const {
    return Extents{
        alpaka::trait::GetExtents<clue::device_buffer<TDev, key_type[]>>{}(m_offsets)[0u],
        alpaka::trait::GetExtents<clue::device_buffer<TDev, mapped_type[]>>{}(m_indexes)[0u]};
  }

  template <concepts::device TDev>
  ALPAKA_FN_HOST inline const auto& AssociationMap<TDev>::indexes() const {
    return m_indexes;
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::iterator AssociationMap<TDev>::begin() {
    return iterator{m_indexes.data()};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::begin() const {
    return const_iterator{m_indexes.data()};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::cbegin() const {
    return const_iterator{m_indexes.data()};
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::iterator AssociationMap<TDev>::end() {
    return iterator{m_indexes.data() + m_offsets[m_nbins]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::end() const {
    return const_iterator{m_indexes.data() + m_offsets[m_nbins]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::cend() const {
    return const_iterator{m_indexes.data() + m_offsets[m_nbins]};
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::size_type AssociationMap<TDev>::count(key_type key) const {
    return m_offsets[key + 1] - m_offsets[key];
  }

  template <concepts::device TDev>
  bool AssociationMap<TDev>::contains(key_type key) const {
    return m_offsets[key + 1] > m_offsets[key];
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::iterator AssociationMap<TDev>::lower_bound(key_type key) {
    return iterator{m_indexes.data() + m_offsets[key]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::lower_bound(key_type key) const {
    return const_iterator{m_indexes.data() + m_offsets[key]};
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::iterator AssociationMap<TDev>::upper_bound(key_type key) {
    return iterator{m_indexes.data() + m_offsets[key + 1]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::upper_bound(key_type key) const {
    return const_iterator{m_indexes.data() + m_offsets[key + 1]};
  }

  template <concepts::device TDev>
  std::pair<typename AssociationMap<TDev>::iterator, typename AssociationMap<TDev>::iterator>
  AssociationMap<TDev>::equal_range(key_type key) {
    return {iterator{m_indexes.data() + m_offsets[key]},
            iterator{m_indexes.data() + m_offsets[key + 1]}};
  }
  template <concepts::device TDev>
  std::pair<typename AssociationMap<TDev>::const_iterator,
            typename AssociationMap<TDev>::const_iterator>
  AssociationMap<TDev>::equal_range(key_type key) const {
    return {const_iterator{m_indexes.data() + m_offsets[key]},
            const_iterator{m_indexes.data() + m_offsets[key + 1]}};
  }

  template <concepts::device TDev>
  inline const AssociationMapView& AssociationMap<TDev>::view() const {
    return m_view;
  }

  template <concepts::device TDev>
  inline AssociationMapView& AssociationMap<TDev>::view() {
    return m_view;
  }

  template <concepts::device TDev>
  template <concepts::queue TQueue>
  inline ALPAKA_FN_HOST void AssociationMap<TDev>::initialize(size_type nelements,
                                                              size_type nbins,
                                                              TQueue& queue) {
    m_indexes = make_device_buffer<int32_t[]>(queue, nelements);
    m_offsets = make_device_buffer<int32_t[]>(queue, nbins);
    alpaka::memset(queue, m_offsets, 0);
    m_nbins = nbins;

    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_nelements = nelements;
    m_view.m_nbins = nbins;
  }

  template <concepts::device TDev>
  template <concepts::queue TQueue>
  inline ALPAKA_FN_HOST void AssociationMap<TDev>::reset(TQueue& queue,
                                                         size_type nelements,
                                                         size_type nbins) {
    alpaka::memset(queue, m_indexes, 0);
    alpaka::memset(queue, m_offsets, 0);
    m_nbins = nbins;

    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_nelements = nelements;
    m_view.m_nbins = nbins;
  }

  template <concepts::device TDev>
  inline auto AssociationMap<TDev>::size() const {
    return m_nbins;
  }

  template <concepts::device TDev>
  ALPAKA_FN_HOST inline auto& AssociationMap<TDev>::indexes() {
    return m_indexes;
  }

  template <concepts::device TDev>
  ALPAKA_FN_ACC inline Span<int32_t> AssociationMap<TDev>::indexes(size_type bin_id) {
    const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
    auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
    return Span<mapped_type>{buf_ptr, size};
  }
  template <concepts::device TDev>
  ALPAKA_FN_HOST inline device_view<TDev, int32_t[]> AssociationMap<TDev>::indexes(
      const TDev& dev, size_type bin_id) {
    const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
    auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
    return make_device_view<int32_t[], TDev>(dev, buf_ptr, size);
  }
  template <concepts::device TDev>
  ALPAKA_FN_ACC inline Span<int32_t> AssociationMap<TDev>::operator[](size_type bin_id) {
    const auto size = m_offsets[bin_id + 1] - m_offsets[bin_id];
    auto* buf_ptr = m_indexes.data() + m_offsets[bin_id];
    return Span<int32_t>{buf_ptr, size};
  }

  template <concepts::device TDev>
  ALPAKA_FN_HOST inline const device_buffer<TDev, int32_t[]>& AssociationMap<TDev>::offsets() const {
    return m_offsets;
  }
  template <concepts::device TDev>
  ALPAKA_FN_HOST inline device_buffer<TDev, int32_t[]>& AssociationMap<TDev>::offsets() {
    return m_offsets;
  }

  template <concepts::device TDev>
  ALPAKA_FN_ACC inline int32_t AssociationMap<TDev>::offsets(size_type bin_id) const {
    return m_offsets[bin_id];
  }

  template <concepts::device TDev>
  template <concepts::accelerator TAcc, typename TFunc, concepts::queue TQueue>
  ALPAKA_FN_HOST inline void AssociationMap<TDev>::fill(size_type size, TFunc func, TQueue& queue) {
    auto bin_buffer = make_device_buffer<int32_t[]>(queue, size);

    // compute associations
    const auto blocksize = 512;
    const auto gridsize = divide_up_by(size, blocksize);
    const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);
    alpaka::exec<TAcc>(
        queue, workdiv, detail::KernelComputeAssociations<TFunc>{}, size, bin_buffer.data(), func);

    auto sizes_buffer = make_device_buffer<int32_t[]>(queue, m_nbins);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelComputeAssociationSizes{},
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
                       detail::KernelFillAssociator{},
                       m_indexes.data(),
                       bin_buffer.data(),
                       temp_offsets.data(),
                       size);
  }

  template <concepts::device TDev>
  template <concepts::accelerator TAcc, concepts::queue TQueue>
  ALPAKA_FN_HOST inline void AssociationMap<TDev>::fill(size_type size,
                                                        std::span<key_type> associations,
                                                        TQueue& queue) {
    const auto blocksize = 512;
    const auto gridsize = divide_up_by(size, blocksize);
    const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);

    auto sizes_buffer = make_device_buffer<key_type[]>(queue, m_nbins);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelComputeAssociationSizes{},
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
                       detail::KernelFillAssociator{},
                       m_indexes.data(),
                       associations.data(),
                       temp_offsets.data(),
                       size);
  }

}  // namespace clue
