#ifndef Points_Alpaka_h
#define Points_Alpaka_h

#include <cstdint>
#include <memory>

#include "../../AlpakaCore/alpakaConfig.hpp"
#include "../../AlpakaCore/alpakaMemory.hpp"
#include "../Points.hpp"

namespace clue {

  class PointsAlpakaView {
  public:
    float* coords;
    float* weight;
    float* rho;
    float* delta;
    int* nearest_higher;
    int* cluster_index;
    int* is_seed;
    int n;
  };

  template <uint8_t Ndim, typename TDev>
	requires alpaka::isDevice<TDev>
  class PointsAlpaka {
  public:
	template <typename TQueue>
    explicit PointsAlpaka(TQueue stream, int n_points)
        : input_buffer{clue::make_device_buffer<float[]>(stream, (Ndim + 3) * n_points)},
          result_buffer{clue::make_device_buffer<int[]>(stream, 3 * n_points)},
          view_dev{clue::make_device_buffer<PointsAlpakaView>(stream)} {
      auto view_host = clue::make_host_buffer<PointsAlpakaView>(stream);
      view_host->coords = input_buffer.data();
      view_host->weight = input_buffer.data() + Ndim * n_points;
      view_host->rho = input_buffer.data() + (Ndim + 1) * n_points;
      view_host->delta = input_buffer.data() + (Ndim + 2) * n_points;
      view_host->nearest_higher = result_buffer.data();
      view_host->cluster_index = result_buffer.data() + n_points;
      view_host->is_seed = result_buffer.data() + 2 * n_points;
      view_host->n = n_points;

      alpaka::memcpy(stream, view_dev, view_host);
    }

    PointsAlpaka(const PointsAlpaka&) = delete;
    PointsAlpaka& operator=(const PointsAlpaka&) = delete;
    PointsAlpaka(PointsAlpaka&&) = default;
    PointsAlpaka& operator=(PointsAlpaka&&) = default;
    ~PointsAlpaka() = default;

    clue::device_buffer<TDev, float[]> input_buffer;
    clue::device_buffer<TDev, int[]> result_buffer;

    PointsAlpakaView* view() { return view_dev.data(); }

  private:
    clue::device_buffer<TDev, PointsAlpakaView> view_dev;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE

#endif
