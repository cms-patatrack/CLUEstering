
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/AssociationMapView.hpp"
#include "CLUEstering/detail/concepts.hpp"
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
        requires(alpaka::Dim<TAcc>::value == 1)
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    size_t size,
                                    int32_t* associations,
                                    TFunc func) const {
        for (auto i : alpaka::uniformElements(acc, size)) {
          associations[i] = func(i);
        }
      }
      template <typename TAcc>
        requires(alpaka::Dim<TAcc>::value == 2)
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    int32_t* associations,
                                    TFunc func,
                                    const auto* event_offsets,
                                    std::size_t max_event_size,
                                    std::size_t /* blocks_per_event */) const {
        for (auto event : alpaka::uniformElementsAlong<0u>(acc)) {
          for (auto local_idx : alpaka::uniformElementsAlong<1u>(acc, max_event_size)) {
            auto global_idx = event_offsets[event] + local_idx;
            if (global_idx < event_offsets[event + 1])
              associations[global_idx] = func(global_idx, event);
          }
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
  inline AssociationMap<TDev>::AssociationMap(size_type nelements, size_type nbins)
    requires std::same_as<TDev, alpaka::DevCpu>
      : m_indexes{make_host_buffer<mapped_type[]>(nelements)},
        m_offsets{make_host_buffer<key_type[]>(nbins + 1)},
        m_view{},
        m_extents{nbins, nelements} {
    if (nelements == 0) {
      throw std::invalid_argument(
          "Number of bins and elements must be positive in AssociationMap constructor");
    }
    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_extents = {nbins, nelements};

    std::memset(m_offsets.data(), 0, (nbins) * sizeof(key_type));
  }

  template <concepts::device TDev>
  template <concepts::queue TQueue>
  inline AssociationMap<TDev>::AssociationMap(size_type nelements, size_type nbins, TQueue& queue)
      : m_indexes{make_device_buffer<mapped_type[]>(queue, nelements)},
        m_offsets{make_device_buffer<key_type[]>(queue, nbins + 1)},
        m_view{},
        m_extents{nbins, nelements} {
    if (nelements == 0) {
      throw std::invalid_argument(
          "Number of bins and elements must be positive in AssociationMap constructor");
    }
    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_extents = {nbins, nelements};

    alpaka::memset(queue, m_indexes, 0);
    alpaka::memset(queue, m_offsets, 0);
  }

  template <concepts::device TDev>
  inline auto AssociationMap<TDev>::extents() const {
    return m_extents;
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
    return iterator{m_indexes.data() + m_offsets[m_extents.keys]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::end() const {
    return const_iterator{m_indexes.data() + m_offsets[m_extents.keys]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::cend() const {
    return const_iterator{m_indexes.data() + m_offsets[m_extents.keys]};
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::size_type AssociationMap<TDev>::count(key_type key) const {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::count.");
    }
    return m_offsets[key + 1] - m_offsets[key];
  }

  template <concepts::device TDev>
  bool AssociationMap<TDev>::contains(key_type key) const {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::contains.");
    }
    return m_offsets[key + 1] > m_offsets[key];
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::iterator AssociationMap<TDev>::lower_bound(key_type key) {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::lower_bound.");
    }
    return iterator{m_indexes.data() + m_offsets[key]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::lower_bound(key_type key) const {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::lower_bound.");
    }
    return const_iterator{m_indexes.data() + m_offsets[key]};
  }

  template <concepts::device TDev>
  AssociationMap<TDev>::iterator AssociationMap<TDev>::upper_bound(key_type key) {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::upper_bound.");
    }
    return iterator{m_indexes.data() + m_offsets[key + 1]};
  }
  template <concepts::device TDev>
  AssociationMap<TDev>::const_iterator AssociationMap<TDev>::upper_bound(key_type key) const {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::upper_bound.");
    }
    return const_iterator{m_indexes.data() + m_offsets[key + 1]};
  }

  template <concepts::device TDev>
  std::pair<typename AssociationMap<TDev>::iterator, typename AssociationMap<TDev>::iterator>
  AssociationMap<TDev>::equal_range(key_type key) {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::equal_range.");
    }
    return {iterator{m_indexes.data() + m_offsets[key]},
            iterator{m_indexes.data() + m_offsets[key + 1]}};
  }
  template <concepts::device TDev>
  std::pair<typename AssociationMap<TDev>::const_iterator,
            typename AssociationMap<TDev>::const_iterator>
  AssociationMap<TDev>::equal_range(key_type key) const {
    if (key < 0 || key >= static_cast<key_type>(m_extents.keys)) {
      throw std::out_of_range("Key out of range in call to AssociationMap::equal_range.");
    }
    return {const_iterator{m_indexes.data() + m_offsets[key]},
            const_iterator{m_indexes.data() + m_offsets[key + 1]}};
  }

  template <concepts::device TDev>
  inline AssociationMap<TDev>::Containers AssociationMap<TDev>::extract() const {
    return Containers{m_offsets, m_indexes};
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
  inline ALPAKA_FN_HOST void AssociationMap<TDev>::initialize(size_type nelements, size_type nbins)
    requires std::same_as<TDev, alpaka::DevCpu>
  {
    m_indexes = make_host_buffer<int32_t[]>(nelements);
    m_offsets = make_host_buffer<int32_t[]>(nbins + 1);
    m_extents = {nbins, nelements};

    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_extents = {nbins, nelements};
  }

  template <concepts::device TDev>
  template <concepts::queue TQueue>
  inline ALPAKA_FN_HOST void AssociationMap<TDev>::initialize(size_type nelements,
                                                              size_type nbins,
                                                              TQueue& queue) {
    m_indexes = make_device_buffer<int32_t[]>(queue, nelements);
    m_offsets = make_device_buffer<int32_t[]>(queue, nbins + 1);
    m_extents = {nbins, nelements};

    m_view.m_indexes = m_indexes.data();
    m_view.m_offsets = m_offsets.data();
    m_view.m_extents = {nbins, nelements};
  }

  template <concepts::device TDev>
  inline ALPAKA_FN_HOST void AssociationMap<TDev>::reset(size_type nelements, size_type nbins) {
    m_extents = {nbins, nelements};
    m_view.m_extents = {nbins, nelements};
  }

  template <concepts::device TDev>
  inline auto AssociationMap<TDev>::size() const {
    return m_extents.keys;
  }

  template <concepts::device TDev>
  ALPAKA_FN_HOST inline auto& AssociationMap<TDev>::indexes() {
    return m_indexes;
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
  template <concepts::accelerator TAcc, typename TFunc, concepts::queue TQueue>
  ALPAKA_FN_HOST inline void AssociationMap<TDev>::fill(size_type size, TFunc func, TQueue& queue) {
    if (m_extents.keys == 0)
      return;

    auto bin_buffer = make_device_buffer<int32_t[]>(queue, size);

    const auto blocksize = 512;
    const auto gridsize = divide_up_by(size, blocksize);
    const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);
    alpaka::exec<TAcc>(
        queue, workdiv, detail::KernelComputeAssociations<TFunc>{}, size, bin_buffer.data(), func);

    auto sizes_buffer = make_device_buffer<int32_t[]>(queue, m_extents.keys);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelComputeAssociationSizes{},
                       bin_buffer.data(),
                       sizes_buffer.data(),
                       size);

    auto block_counter = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, block_counter, 0);

    auto temp_offsets = make_device_buffer<int32_t[]>(queue, m_extents.keys + 1);
    alpaka::memset(queue, temp_offsets, 0u, 1u);
    alpaka::wait(queue);
    internal::algorithm::inclusive_scan(
        sizes_buffer.data(), sizes_buffer.data() + m_extents.keys, temp_offsets.data() + 1);
    alpaka::wait(queue);
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), m_offsets.data(), m_extents.keys + 1),
                   temp_offsets);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelFillAssociator{},
                       m_indexes.data(),
                       bin_buffer.data(),
                       temp_offsets.data(),
                       size);
  }

  template <concepts::device TDev>
  ALPAKA_FN_HOST void AssociationMap<TDev>::fill(std::span<const key_type> associations)
    requires std::same_as<TDev, alpaka::DevCpu>
  {
    std::vector<key_type> sizes(m_extents.keys, 0);
    std::for_each(associations.begin(), associations.end(), [&](key_type key) {
      if (key >= 0) {
        sizes[key]++;
      }
    });

    std::vector<key_type> temporary_keys(m_extents.keys + 1);
    temporary_keys[0] = 0;
    std::inclusive_scan(sizes.begin(), sizes.end(), temporary_keys.begin() + 1);
    std::copy(temporary_keys.data(), temporary_keys.data() + m_extents.keys + 1, m_offsets.data());
    for (auto i = 0u; i < associations.size(); ++i) {
      if (associations[i] >= 0) {
        auto& offset = temporary_keys[associations[i]];
        m_indexes[offset] = i;
        ++offset;
      }
    }
  }

  template <concepts::device TDev>
  template <concepts::accelerator TAcc, concepts::queue TQueue>
  ALPAKA_FN_HOST inline void AssociationMap<TDev>::fill(size_type,
                                                        std::span<const key_type> associations,
                                                        TQueue& queue) {
    if (m_extents.keys == 0)
      return;
    const auto blocksize = 512;
    const auto gridsize = divide_up_by(associations.size(), blocksize);
    const auto workdiv = make_workdiv<TAcc>(gridsize, blocksize);

    auto sizes_buffer = make_device_buffer<key_type[]>(queue, m_extents.keys);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelComputeAssociationSizes{},
                       associations.data(),
                       sizes_buffer.data(),
                       associations.size());

    auto block_counter = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, block_counter, 0);

    auto temp_offsets = make_device_buffer<key_type[]>(queue, m_extents.keys + 1);
    alpaka::memset(queue, temp_offsets, 0u, 1u);

    alpaka::wait(queue);
    internal::algorithm::inclusive_scan(
        sizes_buffer.data(), sizes_buffer.data() + m_extents.keys, temp_offsets.data() + 1);
    alpaka::wait(queue);

    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), m_offsets.data(), m_extents.keys + 1),
                   temp_offsets);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelFillAssociator{},
                       m_indexes.data(),
                       associations.data(),
                       temp_offsets.data(),
                       associations.size());
  }

  template <concepts::device TDev>
  template <concepts::accelerator TAcc, concepts::queue TQueue, typename TFunc>
  ALPAKA_FN_HOST inline void AssociationMap<TDev>::fill_batch(TQueue& queue,
                                                              size_type size,
                                                              TFunc func,
                                                              const auto& event_offsets,
                                                              std::size_t max_event_size) {
    if (m_extents.keys == 0)
      return;

    auto bin_buffer = make_device_buffer<int32_t[]>(queue, size);

    const auto blocksize = 256;
    const auto blocks_per_event = divide_up_by(max_event_size, blocksize);
    const auto batch_size = alpaka::getExtents(event_offsets)[0] - 1;
    const auto batch_workdiv =
        make_workdiv<internal::Acc2D>({batch_size, blocks_per_event}, {1, blocksize});
    alpaka::exec<internal::Acc2D>(queue,
                                  batch_workdiv,
                                  detail::KernelComputeAssociations<TFunc>{},
                                  bin_buffer.data(),
                                  func,
                                  event_offsets.data(),
                                  max_event_size,
                                  blocks_per_event);

    auto sizes_buffer = make_device_buffer<int32_t[]>(queue, m_extents.keys);
    const auto workdiv = make_workdiv<TAcc>(size, blocksize);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelComputeAssociationSizes{},
                       bin_buffer.data(),
                       sizes_buffer.data(),
                       size);

    auto block_counter = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, block_counter, 0);

    auto temp_offsets = make_device_buffer<int32_t[]>(queue, m_extents.keys + 1);
    alpaka::memset(queue, temp_offsets, 0u, 1u);
    alpaka::wait(queue);

    internal::algorithm::inclusive_scan(
        sizes_buffer.data(), sizes_buffer.data() + m_extents.keys, temp_offsets.data() + 1);

    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), m_offsets.data(), m_extents.keys + 1),
                   temp_offsets);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelFillAssociator{},
                       m_indexes.data(),
                       bin_buffer.data(),
                       temp_offsets.data(),
                       size);
  }

}  // namespace clue
