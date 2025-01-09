#ifndef Points_Alpaka_h
#define Points_Alpaka_h

#include <cstdint>
#include <memory>

#include "../../AlpakaCore/alpakaConfig.hpp"
#include "../../AlpakaCore/alpakaMemory.hpp"
#include "../Points.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  template <uint8_t Ndim>
  class PointsAlpaka {
  public:
    PointsAlpaka() = delete;
    explicit PointsAlpaka(Queue stream, int n_points)
        : buffer{clue::make_device_buffer<float[]>(stream, (Ndim + 6) * n_points)},
          view_dev{clue::make_device_buffer<PointsAlpakaView>(stream)} {
      auto view_host = clue::make_host_buffer<PointsAlpakaView>(stream);
      view_host->coords = buffer.data();
      view_host->weight = buffer.data() + Ndim * n_points;
      view_host->rho = buffer.data() + (Ndim + 1) * n_points;
      view_host->delta = buffer.data() + (Ndim + 2) * n_points;
      view_host->nearest_higher =
          reinterpret_cast<int*>(buffer.data() + (Ndim + 3) * n_points);
      view_host->cluster_index =
          reinterpret_cast<int*>(buffer.data() + (Ndim + 4) * n_points);
      view_host->is_seed = reinterpret_cast<int*>(buffer.data() + (Ndim + 5) * n_points);
      view_host->n = n_points;

      alpaka::memcpy(stream, view_dev, view_host);
    }

    PointsAlpaka(const PointsAlpaka&) = delete;
    PointsAlpaka& operator=(const PointsAlpaka&) = delete;
    PointsAlpaka(PointsAlpaka&&) = default;
    PointsAlpaka& operator=(PointsAlpaka&&) = default;
    ~PointsAlpaka() = default;

    clue::device_buffer<Device, float[]> buffer;

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

    PointsAlpakaView* view() { return view_dev.data(); }

  private:
    clue::device_buffer<Device, PointsAlpakaView> view_dev;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE

#endif
