
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

  using namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

  template <typename TFunc>
  struct KernelComputeAssociations {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const int* indexes,
                                  size_t size,
                                  int* associations,
                                  int* nbins,
                                  TFunc func) const {
      auto max = 0;
      for (auto i : alpaka::uniformElements(acc, size)) {
        associations[i] = func(indexes[i]);
        if (associations[i] > max) {
          max = associations[i];
        }
      }
      *nbins = max + 1;
    }
  };

  struct KernelComputeAssociationSizes {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const int* associations,
                                  int* sizes,
                                  size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        alpaka::atomicAdd(acc, &sizes[associations[i]], 1);
      }
    }
  };

  struct KernelFillAssociator {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  int* indexes,
                                  const int* bin_buffer,
                                  int* temp_offsets,
                                  size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        const auto binId = bin_buffer[i];
        const auto position = temp_offsets[binId];
        indexes[position] = i;
        alpaka::atomicAdd(acc, &temp_offsets[binId], 1);
      }
    }
  };

  template <typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class AssociationMap {
  private:
    device_buffer<TDev, int[]> m_indexes;
    device_buffer<TDev, int[]> m_offsets;
    size_t m_size;

  public:
    AssociationMap(size_t size, size_t nbins, const TDev& dev)
        : m_indexes{make_device_buffer<int[]>(dev, size)},
          m_offsets{make_device_buffer<int[]>(dev, nbins)},
          m_size{nbins} {}

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    AssociationMap(size_t size, size_t nbins, const TQueue& queue)
        : m_indexes{make_device_buffer<int[]>(queue, size)},
          m_offsets{make_device_buffer<int[]>(queue, nbins)},
          m_size{nbins} {}

    auto size() const { return m_size; }

    ALPAKA_FN_ACC Span<int> indexes(size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[assoc_id];
      return Span<int>{buf_ptr, size};
    }
    ALPAKA_FN_HOST device_view<TDev, int[]> indexes(const TDev& dev, size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[assoc_id];
      return make_device_view<int[], TDev>(dev, buf_ptr, size);
    }
    ALPAKA_FN_ACC Span<int> operator[](size_t assoc_id) {
      auto size = m_offsets[assoc_id + 1] - m_offsets[assoc_id];
      auto* buf_ptr = m_indexes.data() + m_offsets[assoc_id];
      return Span<int>{buf_ptr, size};
    }


    ALPAKA_FN_HOST device_buffer<TDev, int[]>& offsets() { return m_offsets; }
    ALPAKA_FN_ACC int offsets(size_t assoc_id) const { return m_offsets[assoc_id]; }

    template <typename TFunc>
    ALPAKA_FN_HOST void fill(const int* indexes,
                             size_t size,
                             TFunc func,
                             const TDev& dev) {
      auto nbins_buffer = make_device_buffer<int>(dev);
      auto bin_buffer = make_device_buffer<int[]>(dev, size);

      Queue queue(dev);
      const auto blocksize = 512;
      const auto gridsize = divide_up_by(size, blocksize);
      const auto workdiv = make_workdiv<Acc1D>(gridsize, blocksize);
      alpaka::exec<Acc1D>(queue,
                          workdiv,
                          KernelComputeAssociations<TFunc>{},
                          indexes,
                          size,
                          bin_buffer.data(),
                          nbins_buffer.data(),
                          func);

      int nbins = 0;
      alpaka::memcpy(queue, make_host_view<int>(nbins), nbins_buffer);
      m_size = nbins;
      m_offsets = make_device_buffer<int[]>(dev, nbins + 1);
      auto sizes_buffer = make_device_buffer<int[]>(dev, nbins);
      alpaka::memset(queue, sizes_buffer, 0);
      alpaka::exec<Acc1D>(queue,
                          workdiv,
                          KernelComputeAssociationSizes{},
                          bin_buffer.data(),
                          sizes_buffer.data(),
                          size);

      // prepare for prefix scan
      auto block_counter = make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, block_counter, 0);

      alpaka::memset(queue, m_offsets, 0);

      const auto blocksize_multiblockscan = 1024;
      auto gridsize_multiblockscan = divide_up_by(nbins, blocksize_multiblockscan);
      const auto workdiv_multiblockscan =
          make_workdiv<Acc1D>(gridsize_multiblockscan, blocksize_multiblockscan);
      auto warp_size = alpaka::getPreferredWarpSize(dev);
      alpaka::exec<Acc1D>(queue,
                          workdiv_multiblockscan,
                          multiBlockPrefixScan<int>{},
                          sizes_buffer.data(),
                          m_offsets.data() + 1,
                          nbins,
                          gridsize_multiblockscan,
                          block_counter.data(),
                          warp_size);

      auto temp_offsets = make_device_buffer<int[]>(queue, nbins + 1);
      alpaka::memcpy(queue, temp_offsets, m_offsets);
      alpaka::exec<Acc1D>(queue,
                          workdiv,
                          KernelFillAssociator{},
                          this->view(),
                          bin_buffer.data(),
                          temp_offsets.data(),
                          size);
    }
  };

  template <typename TFunc,
            typename TDev,
            typename = std::enable_if_t<alpaka::isDevice<TDev>>>

  ALPAKA_FN_HOST AssociationMap<TDev> CreateAssociationMap(const int* indexes,
                                                           size_t size,
                                                           TFunc func,
                                                           const TDev& dev) {
    auto nbins_buffer = make_device_buffer<int>(dev);
    auto bin_buffer = make_device_buffer<int[]>(dev, size);

    Queue queue(dev);
    const auto blocksize = 512;
    const auto gridsize = divide_up_by(size, blocksize);
    const auto workdiv = make_workdiv<Acc1D>(gridsize, blocksize);
    alpaka::exec<Acc1D>(queue,
                        workdiv,
                        KernelComputeAssociations<TFunc>{},
                        indexes,
                        size,
                        bin_buffer.data(),
                        nbins_buffer.data(),
                        func);

    auto nbins = *nbins_buffer.data();
    auto sizes_buffer = make_device_buffer<int[]>(dev, nbins);
    alpaka::memset(queue, sizes_buffer, 0);
    alpaka::exec<Acc1D>(queue,
                        workdiv,
                        KernelComputeAssociationSizes{},
                        bin_buffer.data(),
                        sizes_buffer.data(),
                        size);

    AssociationMap<TDev> assoc_map(size, nbins + 1, dev);

    // prepare for prefix scan
    auto block_counter = make_device_buffer<int32_t>(queue);
    alpaka::memset(queue, block_counter, 0);
    alpaka::memset(queue, assoc_map.offsets(), 0);

    const auto blocksize_multiblockscan = 1;
    auto gridsize_multiblockscan = divide_up_by(nbins, blocksize_multiblockscan);
    const auto workdiv_multiblockscan =
        make_workdiv<Acc1D>(gridsize_multiblockscan, blocksize_multiblockscan);
    auto warp_size = alpaka::getPreferredWarpSize(dev);
    alpaka::exec<Acc1D>(queue,
                        workdiv_multiblockscan,
                        multiBlockPrefixScan<int>{},
                        sizes_buffer.data(),
                        assoc_map.offsets().data() + 1,
                        nbins,
                        gridsize_multiblockscan,
                        block_counter.data(),
                        warp_size);

    auto temp_offsets = make_device_buffer<int[]>(queue, nbins + 1);
    alpaka::memcpy(queue, temp_offsets, assoc_map.offsets());
    alpaka::exec<Acc1D>(queue,
                        workdiv,
                        KernelFillAssociator{},
                        assoc_map.view(),
                        bin_buffer.data(),
                        temp_offsets.data(),
                        size);

    return assoc_map;
  }

}  // namespace clue
